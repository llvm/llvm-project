//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Address space layout of a module's PT_LOAD segments
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_LOAD_LAYOUT_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_LOAD_LAYOUT_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/elf/elf_types.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// How much address space a module needs, and where its segments sit in it.
//
// A loader has to reserve the whole span in one mapping first and then map the
// segments into it. Mapping each segment separately would let something else
// take an address in the middle of the range, and the segments have to keep
// their relative offsets for the module's own relocations to be correct.
class LoadLayout {
public:
  LIBC_INLINE constexpr LoadLayout() = default;

  LIBC_INLINE LoadLayout(const ElfW(Phdr) * phdrs, ElfW(Half) phnum) {
    bool seen = false;
    for (ElfW(Half) i = 0; i < phnum; ++i) {
      if (phdrs[i].p_type != PT_LOAD)
        continue;
      const ElfW(Addr) start = phdrs[i].p_vaddr;
      const ElfW(Addr) end = phdrs[i].p_vaddr + phdrs[i].p_memsz;
      if (!seen) {
        first_ = start;
        last_ = end;
        seen = true;
      } else {
        if (start < first_)
          first_ = start;
        if (end > last_)
          last_ = end;
      }
      if (phdrs[i].p_align > align_)
        align_ = phdrs[i].p_align;
      ++count_;
    }
  }

  LIBC_INLINE constexpr bool empty() const { return count_ == 0; }
  LIBC_INLINE constexpr size_t segment_count() const { return count_; }

  // Lowest and highest link time addresses the module occupies. A shared
  // object normally starts at zero, an executable at its fixed base.
  LIBC_INLINE constexpr ElfW(Addr) first_vaddr() const { return first_; }
  LIBC_INLINE constexpr ElfW(Addr) last_vaddr() const { return last_; }

  // Bytes of address space to reserve, from the start of the first segment's
  // page to the end of the last.
  LIBC_INLINE size_t reservation_size(size_t page_size) const {
    if (count_ == 0)
      return 0;
    return page_align_up(last_, page_size) - page_align_down(first_, page_size);
  }

  LIBC_INLINE constexpr Xword alignment() const { return align_; }

  LIBC_INLINE static ElfW(Addr)
      page_align_down(ElfW(Addr) value, size_t page_size) {
    return value & ~static_cast<ElfW(Addr)>(page_size - 1);
  }

  LIBC_INLINE static ElfW(Addr)
      page_align_up(ElfW(Addr) value, size_t page_size) {
    return page_align_down(value + page_size - 1, page_size);
  }

private:
  ElfW(Addr) first_ = 0;
  ElfW(Addr) last_ = 0;
  Xword align_ = 0;
  size_t count_ = 0;
};

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_LOAD_LAYOUT_H
