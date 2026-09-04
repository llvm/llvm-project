//===-- Definition of macros from elf.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_MACROS_ELF_MACROS_H
#define LLVM_LIBC_MACROS_ELF_MACROS_H

// Symbol Table

#define ELF32_ST_BIND(i) ((i) >> 4)
#define ELF32_ST_TYPE(i) ((i) & 0xf)
#define ELF32_ST_INFO(b, t) (((b) << 4) + ((t) & 0xf))

#define ELF64_ST_BIND(i) ((i) >> 4)
#define ELF64_ST_TYPE(i) ((i) & 0xf)
#define ELF64_ST_INFO(b, t) (((b) << 4) + ((t) & 0xf))

#define ELF32_ST_VISIBILITY(o) ((o) & 0x3)
#define ELF64_ST_VISIBILITY(o) ((o) & 0x3)

// Relocations

#define ELF32_R_SYM(i) ((i) >> 8)
#define ELF32_R_TYPE(i) ((unsigned char)(i))
#define ELF32_R_INFO(s, t) (((s) << 8) + (unsigned char)(t))

#define ELF64_R_SYM(i) ((i) >> 32)
#define ELF64_R_TYPE(i) ((i) & 0xffffffffL)
#define ELF64_R_INFO(s, t) (((s) << 32) + ((t) & 0xffffffffL))

// Per architecture relocation types.
//
// Only the types the startup code and the dynamic loader apply are listed.
// RELATIVE adds the load bias to the addend; IRELATIVE additionally calls the
// resolver at that address and stores what it returns.

// x86_64
#define R_X86_64_RELATIVE 8
#define R_X86_64_GLOB_DAT 6
#define R_X86_64_JUMP_SLOT 7
#define R_X86_64_IRELATIVE 37

// aarch64
#define R_AARCH64_RELATIVE 1027
#define R_AARCH64_GLOB_DAT 1025
#define R_AARCH64_JUMP_SLOT 1026
#define R_AARCH64_IRELATIVE 1032
#define R_AARCH64_COPY 1024
#define R_AARCH64_TLS_DTPMOD 1028
#define R_AARCH64_TLS_DTPREL 1029

// arm
#define R_ARM_RELATIVE 23
#define R_ARM_GLOB_DAT 21
#define R_ARM_JUMP_SLOT 22
#define R_ARM_IRELATIVE 160
#define R_ARM_COPY 20

// riscv
#define R_RISCV_RELATIVE 3
#define R_RISCV_JUMP_SLOT 5
#define R_RISCV_IRELATIVE 58
#define R_RISCV_COPY 4
#define R_RISCV_TLS_DTPMOD64 7
#define R_RISCV_TLS_DTPREL64 9

#endif // LLVM_LIBC_MACROS_ELF_MACROS_H
