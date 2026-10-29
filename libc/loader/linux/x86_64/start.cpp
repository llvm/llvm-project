//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Entry stub of the startup dynamic linker (x86_64)
///
//===----------------------------------------------------------------------===//

#include "loader/linux/loader_main.h"

#include "hdr/link_macros.h"
#include "src/__support/elf/self_relocate.h"
#include "src/__support/macros/config.h"

// Supplied by the linker: the loader's own dynamic section and the address it
// was actually mapped at.
extern "C" const ElfW(Dyn) _DYNAMIC[];
extern "C" const char __ehdr_start[];

namespace LIBC_NAMESPACE_DECL {

// Runs before the loader has been relocated, so it may not read a global or
// call anything through the GOT. Both symbols it needs are resolved
// pc-relatively by the linker, which works without any relocation applied.
extern "C" [[gnu::used]] uintptr_t __libc_loader_start(void *stack_pointer) {
  const ElfW(Addr) load_bias = reinterpret_cast<ElfW(Addr)>(__ehdr_start);
  elf::self_relocate(_DYNAMIC, load_bias);
  // Everything from here on has a usable address.
  return loader_link(stack_pointer);
}

} // namespace LIBC_NAMESPACE_DECL

// The kernel enters here with the stack pointer at argc and nothing else set
// up. The stack must reach the program's entry point untouched, so the
// pointer is passed along and then restored before the jump.
__asm__(".text\n"
        ".globl _start\n"
        ".type _start,@function\n"
        "_start:\n"
        "  xor %rbp, %rbp\n"
        "  mov %rsp, %rdi\n" // the stack is the only argument
        "  and $-16, %rsp\n" // the ABI wants it aligned for the call
        "  push %rdi\n"
        "  push %rdi\n" // keep the alignment
        "  call __libc_loader_start\n"
        "  pop %rdi\n"
        "  pop %rsp\n"  // hand the program the original stack
        "  jmp *%rax\n" // and the entry point it returned
        ".size _start,.-_start\n");
