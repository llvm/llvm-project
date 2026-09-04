//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Entry point of the startup dynamic linker
///
//===----------------------------------------------------------------------===//

#include "loader/linux/loader_main.h"

#include "src/__support/OSUtil/io.h"
#include "src/__support/OSUtil/syscall.h"

#include "src/__support/elf/startup_linker.h"
#include "src/__support/elf/startup_stack.h"
#include "src/__support/macros/config.h"
#include <sys/syscall.h>

namespace LIBC_NAMESPACE_DECL {

namespace {

// The kernel always supplies AT_PAGESZ, but a sane value costs nothing.
constexpr size_t FALLBACK_PAGE_SIZE = 4096;

// The loader deliberately depends on nothing outside itself, so this is the
// syscall rather than libc's exit: there is no libc yet.
[[noreturn]] void die(const char *message) {
  write_to_stderr(message);
  for (;;)
    syscall_impl<long>(SYS_exit_group, 127);
}

} // anonymous namespace

uintptr_t loader_link(void *stack_pointer) {
  elf::StartupStack stack = elf::StartupStack::from(stack_pointer);

  auto image = elf::executable_from(stack);
  if (!image)
    die("loader: the kernel did not describe the executable\n");

  // A program run directly rather than through this loader has no entry for
  // us to hand control to.
  if (image->entry == 0)
    die("loader: no entry point\n");

  const size_t page_size =
      static_cast<size_t>(stack.auxval(AT_PAGESZ).value_or(FALLBACK_PAGE_SIZE));

  elf::StartupLinker linker(page_size, stack.envp());
  if (!linker.link(stack, *image))
    die("loader: linking failed\n");

  return image->entry;
}

} // namespace LIBC_NAMESPACE_DECL
