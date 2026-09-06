//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Symbol lookup in a loaded ELF module
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_SYMBOL_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_SYMBOL_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/elf/hash.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// The header of a DT_GNU_HASH table, followed by the Bloom filter, the bucket
// array and the hash array.
struct GnuHashHeader {
  uint32_t nbuckets;
  // Symbols below this index are not in the hash table at all.
  uint32_t symoffset;
  uint32_t bloom_size;
  uint32_t bloom_shift;
};

// Looks names up in one module's .dynsym using its DT_GNU_HASH table.
//
// The Bloom filter is checked first. It can only give a false positive, never
// a false negative, so a miss there is conclusive and skips the chain walk
// entirely. That is the whole reason GNU hash is faster than DT_HASH.
class GnuSymbolTable {
public:
  LIBC_INLINE constexpr GnuSymbolTable() = default;
  LIBC_INLINE GnuSymbolTable(const void *hash_table, const ElfW(Sym) * symtab,
                             const char *strtab)
      : symtab_(symtab), strtab_(strtab) {
    if (hash_table == nullptr)
      return;
    header_ = static_cast<const GnuHashHeader *>(hash_table);
    bloom_ = reinterpret_cast<const ElfW(Addr) *>(header_ + 1);
    buckets_ = reinterpret_cast<const uint32_t *>(bloom_ + header_->bloom_size);
    chain_ = buckets_ + header_->nbuckets;
  }

  LIBC_INLINE constexpr bool empty() const { return header_ == nullptr; }

  // Returns the matching symbol, or nullptr. Defined only for names this
  // module actually exports; callers walk modules in search order themselves.
  LIBC_INLINE const ElfW(Sym) * lookup(const char *name) const {
    if (header_ == nullptr || header_->nbuckets == 0)
      return nullptr;

    const uint32_t hash = gnu_hash(name);
    if (!bloom_may_contain(hash))
      return nullptr;

    uint32_t index = buckets_[hash % header_->nbuckets];
    if (index < header_->symoffset)
      return nullptr;

    for (;; ++index) {
      const uint32_t entry = chain_[index - header_->symoffset];
      // The low bit is the end-of-chain marker, so it is masked out of the
      // comparison rather than being part of the hash.
      if ((entry | 1) == (hash | 1) && matches(index, name))
        return &symtab_[index];
      if (entry & 1)
        return nullptr;
    }
  }

  // How many symbols .dynsym holds. The section header is not mapped, so the
  // count is worked out from the hash table: every symbol at or above
  // symoffset is on exactly one chain, and the chains end at the highest
  // index there is.
  LIBC_INLINE uint32_t count() const {
    if (header_ == nullptr)
      return 0;
    uint32_t last = header_->symoffset;
    for (uint32_t i = 0; i < header_->nbuckets; ++i)
      if (buckets_[i] > last)
        last = buckets_[i];
    if (last < header_->symoffset)
      return header_->symoffset;
    // Walk to the end of the chain the highest bucket starts.
    while ((chain_[last - header_->symoffset] & 1) == 0)
      ++last;
    return last + 1;
  }

  LIBC_INLINE const ElfW(Sym) * symbol(uint32_t index) const {
    return symtab_ == nullptr ? nullptr : &symtab_[index];
  }

  LIBC_INLINE const char *name(uint32_t index) const {
    if (symtab_ == nullptr || strtab_ == nullptr)
      return nullptr;
    return strtab_ + symtab_[index].st_name;
  }

private:
  LIBC_INLINE bool bloom_may_contain(uint32_t hash) const {
    if (header_->bloom_size == 0)
      return true;
    constexpr uint32_t BITS = 8 * sizeof(ElfW(Addr));
    const ElfW(Addr) word = bloom_[(hash / BITS) % header_->bloom_size];
    const ElfW(Addr) mask = (ElfW(Addr))1 << (hash % BITS) |
                            (ElfW(Addr))1
                                << ((hash >> header_->bloom_shift) % BITS);
    return (word & mask) == mask;
  }

  LIBC_INLINE bool matches(uint32_t index, const char *name) const {
    const char *candidate = strtab_ + symtab_[index].st_name;
    for (; *candidate == *name; ++candidate, ++name)
      if (*candidate == '\0')
        return true;
    return false;
  }

  const GnuHashHeader *header_ = nullptr;
  const ElfW(Addr) *bloom_ = nullptr;
  const uint32_t *buckets_ = nullptr;
  const uint32_t *chain_ = nullptr;
  const ElfW(Sym) *symtab_ = nullptr;
  const char *strtab_ = nullptr;
};

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_SYMBOL_H
