//===-- assembly.h - libc assembler support macros based on compiler-rt's -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_ASSEMBLY_H
#define LLVM_LIBC_SRC___SUPPORT_ASSEMBLY_H

#ifndef __ASSEMBLER__
#error "No not include assembly.h in non-asm sources"
#endif

#define GLUE2_(a, b) a##b
#define GLUE(a, b) GLUE2_(a, b)
#define SYMBOL_NAME(name) GLUE(__USER_LABEL_PREFIX__, name)

#if defined(__ELF__) && (defined(__GNU__) || defined(__FreeBSD__) ||           \
                         defined(__Fuchsia__) || defined(__linux__))

// clang-format off
#define NO_EXEC_STACK_DIRECTIVE .section .note.GNU-stack, "", @progbits
#define SYMBOL_IS_FUNC(name) .type SYMBOL_NAME(name), %function
#define END_FUNC(name) .size SYMBOL_NAME(name), . - SYMBOL_NAME(name)
// clang-format on

#else // !ELF
#define NO_EXEC_STACK_DIRECTIVE
#define SYMBOL_IS_FUNC(name)
#define END_FUNC(name)
#endif // ELF

#endif // LLVM_LIBC_SRC___SUPPORT_ASSEMBLY_H
