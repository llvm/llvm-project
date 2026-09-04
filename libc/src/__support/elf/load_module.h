//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Loading a shared object from a file
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_LOAD_MODULE_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_LOAD_MODULE_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/lseek.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/read.h"
#include "src/__support/elf/map_segments.h"
#include "src/__support/elf/module.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// A shared object that has been mapped and had its own relocations applied.
//
// It is not linked against anything yet: symbols it imports from other modules
// still need binding, which needs the rest of the module set.
struct LoadedModule {
  MappedModule mapping;
  Module module;
};

#ifdef __LP64__
constexpr unsigned char ELFCLASS_NATIVE = ELFCLASS64;
#else
constexpr unsigned char ELFCLASS_NATIVE = ELFCLASS32;
#endif

// The largest program header table this will read, so a malformed file cannot
// ask for an unbounded stack buffer.
constexpr ElfW(Half) MAX_PHNUM = 128;

LIBC_INLINE bool is_loadable_elf(const ElfW(Ehdr) & header) {
  return header.e_ident[EI_MAG0] == ELFMAG0 &&
         header.e_ident[EI_MAG1] == ELFMAG1 &&
         header.e_ident[EI_MAG2] == ELFMAG2 &&
         header.e_ident[EI_MAG3] == ELFMAG3 &&
         header.e_ident[EI_CLASS] == ELFCLASS_NATIVE &&
         (header.e_type == ET_DYN || header.e_type == ET_EXEC) &&
         header.e_phentsize == sizeof(ElfW(Phdr)) && header.e_phnum != 0 &&
         header.e_phnum <= MAX_PHNUM;
}

// Opens `path`, maps it, and applies the relocations it can resolve on its
// own. The caller owns the mapping and releases it with unmap_module.
LIBC_INLINE ErrorOr<LoadedModule> load_module(const char *path,
                                              size_t page_size) {
  auto fd = linux_syscalls::open(path, O_RDONLY, 0);
  if (!fd.has_value())
    return Error(fd.error());

  auto close_fd = [&] { linux_syscalls::close(fd.value()); };

  ElfW(Ehdr) header;
  auto header_read = linux_syscalls::read(fd.value(), &header, sizeof(header));
  if (!header_read.has_value() ||
      static_cast<size_t>(header_read.value()) != sizeof(header)) {
    close_fd();
    return Error(ENOEXEC);
  }
  if (!is_loadable_elf(header)) {
    close_fd();
    return Error(ENOEXEC);
  }

  ElfW(Phdr) phdrs[MAX_PHNUM];
  const size_t phdrs_size = header.e_phnum * sizeof(ElfW(Phdr));
  if (!linux_syscalls::lseek(fd.value(), static_cast<off_t>(header.e_phoff),
                             SEEK_SET)
           .has_value()) {
    close_fd();
    return Error(ENOEXEC);
  }
  auto phdrs_read = linux_syscalls::read(fd.value(), phdrs, phdrs_size);
  if (!phdrs_read.has_value() ||
      static_cast<size_t>(phdrs_read.value()) != phdrs_size) {
    close_fd();
    return Error(ENOEXEC);
  }

  auto mapped =
      map_segments(fd.value(), phdrs, header.e_phnum, page_size, path);
  close_fd();
  if (!mapped.has_value())
    return Error(mapped.error());

  LoadedModule loaded;
  loaded.mapping = mapped.value();

  // Describe the module through its mapped program headers rather than the
  // copy read from the file, so the description stays valid for as long as
  // the mapping does.
  const ElfW(Phdr) *mapped_phdrs = reinterpret_cast<const ElfW(Phdr) *>(
      loaded.mapping.load_bias + header.e_phoff);
  loaded.module = Module(mapped_phdrs, header.e_phnum, loaded.mapping.load_bias,
                         loaded.mapping.name);

  // Relative relocations are the ones a module can apply to itself. Anything
  // that names a symbol needs the other modules to be loaded first.
  loaded.module.relocations().apply_relative(loaded.mapping.load_bias);
  return loaded;
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_LOAD_MODULE_H
