//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Reader for a module's PT_DYNAMIC array
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_DYNAMIC_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_DYNAMIC_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/elf/elf_types.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// A view over the DT_NULL terminated array a PT_DYNAMIC segment points at.
//
// Tag values are either a plain integer or an address. An address is stored
// relative to how the module was linked, so it has to be adjusted by the load
// bias to be usable at runtime. The two accessors keep that distinction
// explicit, since applying the bias to an integer is a common way to get a
// loader subtly wrong.
class DynamicTable {
public:
  LIBC_INLINE constexpr DynamicTable() = default;
  LIBC_INLINE constexpr DynamicTable(const ElfW(Dyn) * entries,
                                     ElfW(Addr) load_bias)
      : entries_(entries), load_bias_(load_bias) {}

  LIBC_INLINE constexpr bool empty() const { return entries_ == nullptr; }
  LIBC_INLINE constexpr ElfW(Addr) load_bias() const { return load_bias_; }

  // The d_un.d_val of the first entry with this tag.
  LIBC_INLINE cpp::optional<Xword> value(Sxword tag) const {
    if (const ElfW(Dyn) *entry = find(tag))
      return entry->d_un.d_val;
    return cpp::nullopt;
  }

  // The d_un.d_ptr of the first entry with this tag, adjusted by the load
  // bias so that it is valid in this process.
  LIBC_INLINE cpp::optional<ElfW(Addr)> address(Sxword tag) const {
    if (const ElfW(Dyn) *entry = find(tag))
      return entry->d_un.d_ptr + load_bias_;
    return cpp::nullopt;
  }

  LIBC_INLINE bool contains(Sxword tag) const { return find(tag) != nullptr; }

  // Calls callback(const ElfW(Dyn) &) for every entry with this tag. DT_NEEDED
  // is the reason this exists: a module may carry any number of them.
  template <typename F>
  LIBC_INLINE void for_each(Sxword tag, F callback) const {
    if (entries_ == nullptr)
      return;
    for (const ElfW(Dyn) *entry = entries_; entry->d_tag != DT_NULL; ++entry)
      if (entry->d_tag == tag)
        callback(*entry);
  }

  LIBC_INLINE size_t count(Sxword tag) const {
    size_t found = 0;
    for_each(tag, [&found](const ElfW(Dyn) &) { ++found; });
    return found;
  }

private:
  LIBC_INLINE const ElfW(Dyn) * find(Sxword tag) const {
    if (entries_ == nullptr)
      return nullptr;
    for (const ElfW(Dyn) *entry = entries_; entry->d_tag != DT_NULL; ++entry)
      if (entry->d_tag == tag)
        return entry;
    return nullptr;
  }

  const ElfW(Dyn) *entries_ = nullptr;
  ElfW(Addr) load_bias_ = 0;
};

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_DYNAMIC_H
