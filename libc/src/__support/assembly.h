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

#if defined(ELF) && (defined(__GNU__) || defined(__FreeBSD__) ||               \
                     defined(__Fuchsia__) || defined(__linux__))
#define NO_EXEC_STACK_DIRECTIVE .section.note.GNU - stack, "", % progbits
#else
#define NO_EXEC_STACK_DIRECTIVE
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_ASSEMBLY_H
