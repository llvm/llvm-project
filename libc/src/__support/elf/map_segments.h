//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Mapping a module's PT_LOAD segments into memory
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_MAP_SEGMENTS_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_MAP_SEGMENTS_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/errno_macros.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/sys_mman_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/mmap.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/munmap.h"
#include "src/__support/elf/load_layout.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// A module that has been mapped into this process.
struct MappedModule {
  void *reservation = nullptr;
  size_t reservation_size = 0;
  ElfW(Addr) load_bias = 0;
  // The path the module was loaded from, held inside the reservation so that
  // it lasts exactly as long as the module does. Null where the path was too
  // long to keep.
  const char *name = nullptr;
};

LIBC_INLINE int segment_protection(ElfW(Word) flags) {
  int prot = 0;
  if (flags & PF_R)
    prot |= PROT_READ;
  if (flags & PF_W)
    prot |= PROT_WRITE;
  if (flags & PF_X)
    prot |= PROT_EXEC;
  return prot;
}

// Maps every PT_LOAD segment of an open module into one contiguous region.
//
// The whole span is reserved first with a single PROT_NONE mapping and the
// segments are then placed inside it. Mapping them one at a time would let
// something else take an address in the middle of the range, and the segments
// have to keep their relative offsets for the module's own relocations to be
// correct.
//
// A copy of `path` is kept in a page of its own past the segments. The string
// a caller passes in is often a buffer it is about to reuse or release, and
// the module has to be able to say where it came from for as long as it is
// loaded.
//
// On failure the reservation is released, so a caller never has to unmap a
// partially built module.
LIBC_INLINE ErrorOr<MappedModule> map_segments(int fd, const ElfW(Phdr) * phdrs,
                                               ElfW(Half) phnum,
                                               size_t page_size,
                                               const char *path) {
  LoadLayout layout(phdrs, phnum);
  if (layout.empty())
    return Error(ENOEXEC);

  const size_t span = layout.reservation_size(page_size);
  const size_t total = span + page_size;
  auto reservation = linux_syscalls::mmap(nullptr, total, PROT_NONE,
                                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (!reservation.has_value())
    return Error(reservation.error());

  const ElfW(Addr) base = reinterpret_cast<ElfW(Addr)>(reservation.value());
  const ElfW(Addr) first =
      LoadLayout::page_align_down(layout.first_vaddr(), page_size);
  const ElfW(Addr) bias = base - first;

  auto fail = [&](int error) -> ErrorOr<MappedModule> {
    linux_syscalls::munmap(reservation.value(), total);
    return Error(error);
  };

  // How far the segments mapped so far reach, so that a segment which shares
  // a page with the one before it does not map over what is already there.
  ElfW(Addr) mapped_to = 0;

  for (ElfW(Half) i = 0; i < phnum; ++i) {
    const ElfW(Phdr) &phdr = phdrs[i];
    if (phdr.p_type != PT_LOAD || phdr.p_memsz == 0)
      continue;

    const int prot = segment_protection(phdr.p_flags);
    const ElfW(Addr) start =
        LoadLayout::page_align_down(bias + phdr.p_vaddr, page_size);
    // The file offset has to be aligned by the same amount the address was.
    const ElfW(Off) offset =
        phdr.p_offset -
        (phdr.p_vaddr - LoadLayout::page_align_down(phdr.p_vaddr, page_size));
    const ElfW(Addr) file_end = bias + phdr.p_vaddr + phdr.p_filesz;

    if (phdr.p_filesz != 0) {
      const size_t length =
          LoadLayout::page_align_up(file_end, page_size) - start;
      auto mapped = linux_syscalls::mmap(reinterpret_cast<void *>(start),
                                         length, prot, MAP_PRIVATE | MAP_FIXED,
                                         fd, static_cast<off_t>(offset));
      if (!mapped.has_value())
        return fail(mapped.error());
      mapped_to = start + length;
    }

    if (phdr.p_memsz <= phdr.p_filesz)
      continue;

    // Anything past p_filesz is .bss. The remainder of the last file backed
    // page still holds whatever the file had there, so it is zeroed, and any
    // whole pages beyond it are mapped anonymously.
    if (phdr.p_filesz != 0 && (prot & PROT_WRITE) != 0) {
      const ElfW(Addr) page_end =
          LoadLayout::page_align_up(file_end, page_size);
      for (ElfW(Addr) p = file_end; p < page_end; ++p)
        *reinterpret_cast<unsigned char *>(p) = 0;
    }

    // A segment with nothing in the file behind it is all .bss, and the page
    // its first byte falls in is its own to map: rounding up from where the
    // file part ended would skip that page and leave the segment with no
    // memory at all. Where an earlier segment already reached into that page,
    // the mapping starts after it instead of over it.
    ElfW(Addr) bss_start = phdr.p_filesz != 0
                               ? LoadLayout::page_align_up(file_end, page_size)
                               : start;
    if (bss_start < mapped_to)
      bss_start = mapped_to;
    const ElfW(Addr) bss_end = LoadLayout::page_align_up(
        bias + phdr.p_vaddr + phdr.p_memsz, page_size);
    if (bss_end > bss_start) {
      auto zeroed = linux_syscalls::mmap(
          reinterpret_cast<void *>(bss_start), bss_end - bss_start, prot,
          MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
      if (!zeroed.has_value())
        return fail(zeroed.error());
      mapped_to = bss_end;
    }
  }

  auto name_page = linux_syscalls::mmap(
      reinterpret_cast<void *>(base + span), page_size, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
  if (!name_page.has_value())
    return fail(name_page.error());

  const char *stored = nullptr;
  if (path != nullptr) {
    char *out = reinterpret_cast<char *>(base + span);
    size_t i = 0;
    for (; path[i] != '\0' && i + 1 < page_size; ++i)
      out[i] = path[i];
    out[i] = '\0';
    // A path with no room to be kept whole is better left unnamed than
    // truncated: a truncated name could compare equal to another module's.
    if (path[i] == '\0')
      stored = out;
  }

  MappedModule module;
  module.reservation = reservation.value();
  module.reservation_size = total;
  module.load_bias = bias;
  module.name = stored;
  return module;
}

LIBC_INLINE void unmap_module(const MappedModule &module) {
  if (module.reservation != nullptr)
    linux_syscalls::munmap(module.reservation, module.reservation_size);
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_MAP_SEGMENTS_H
