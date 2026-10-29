//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The stack the kernel hands a new process
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_STARTUP_STACK_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_STARTUP_STACK_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// The layout at the stack pointer when the kernel starts a process:
//
//   argc, argv[0..argc-1], NULL, envp[0..], NULL, auxv[0..], AT_NULL
//
// This is the first thing a loader entry point reads, before anything else is
// available. It is a view, not a copy: everything points into that stack.
class StartupStack {
public:
  LIBC_INLINE constexpr StartupStack() = default;

  LIBC_INLINE static StartupStack from(void *stack_pointer) {
    StartupStack result;
    auto *words = reinterpret_cast<uintptr_t *>(stack_pointer);
    result.argc_ = words[0];
    result.argv_ = reinterpret_cast<char **>(&words[1]);
    // envp begins one past argv's terminating null.
    char **cursor = result.argv_ + result.argc_;
    result.envp_ = cursor + 1;
    cursor = result.envp_;
    while (*cursor != nullptr)
      ++cursor;
    result.auxv_ = reinterpret_cast<ElfW(auxv_t) *>(cursor + 1);
    return result;
  }

  LIBC_INLINE constexpr uintptr_t argc() const { return argc_; }
  LIBC_INLINE constexpr char **argv() const { return argv_; }
  LIBC_INLINE constexpr char **envp() const { return envp_; }
  LIBC_INLINE constexpr ElfW(auxv_t) * auxv() const { return auxv_; }

  // The value of one auxiliary vector entry, if the kernel supplied it.
  LIBC_INLINE cpp::optional<unsigned long> auxval(unsigned long type) const {
    if (auxv_ == nullptr)
      return cpp::nullopt;
    for (ElfW(auxv_t) *entry = auxv_; entry->a_type != AT_NULL; ++entry)
      if (entry->a_type == type)
        return entry->a_un.a_val;
    return cpp::nullopt;
  }

  LIBC_INLINE size_t envc() const {
    size_t count = 0;
    if (envp_ != nullptr)
      for (char **e = envp_; *e != nullptr; ++e)
        ++count;
    return count;
  }

private:
  uintptr_t argc_ = 0;
  char **argv_ = nullptr;
  char **envp_ = nullptr;
  ElfW(auxv_t) *auxv_ = nullptr;
};

// Where the kernel put the main executable, taken from the auxiliary vector.
//
// A loader needs this to describe the executable without reading it back off
// disk: the kernel has already mapped it and says where its program headers
// landed.
struct ExecutableImage {
  const ElfW(Phdr) *phdrs = nullptr;
  ElfW(Half) phnum = 0;
  ElfW(Addr) entry = 0;
  // Where the interpreter itself was mapped. Zero when the program was
  // started directly rather than through an interpreter.
  ElfW(Addr) interpreter_base = 0;
};

LIBC_INLINE cpp::optional<ExecutableImage>
executable_from(const StartupStack &stack) {
  auto phdr = stack.auxval(AT_PHDR);
  auto phnum = stack.auxval(AT_PHNUM);
  auto phent = stack.auxval(AT_PHENT);
  if (!phdr || !phnum || !phent)
    return cpp::nullopt;
  // A mismatched entry size means the vector does not describe headers this
  // build understands, and walking them would read past each one.
  if (*phent != sizeof(ElfW(Phdr)))
    return cpp::nullopt;

  ExecutableImage image;
  image.phdrs = reinterpret_cast<const ElfW(Phdr) *>(*phdr);
  image.phnum = static_cast<ElfW(Half)>(*phnum);
  image.entry = static_cast<ElfW(Addr)>(stack.auxval(AT_ENTRY).value_or(0));
  image.interpreter_base =
      static_cast<ElfW(Addr)>(stack.auxval(AT_BASE).value_or(0));
  return image;
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_STARTUP_STACK_H
