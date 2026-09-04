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

// aarch64
#define R_AARCH64_ABS64 257
#define R_AARCH64_RELATIVE 1027
#define R_AARCH64_GLOB_DAT 1025
#define R_AARCH64_JUMP_SLOT 1026
#define R_AARCH64_IRELATIVE 1032
#define R_AARCH64_COPY 1024
#define R_AARCH64_TLS_DTPMOD 1028
#define R_AARCH64_TLS_DTPREL 1029

// arm
#define R_ARM_ABS32 2
#define R_ARM_RELATIVE 23
#define R_ARM_GLOB_DAT 21
#define R_ARM_JUMP_SLOT 22
#define R_ARM_IRELATIVE 160
#define R_ARM_COPY 20

// riscv
#define R_RISCV_64 2
#define R_RISCV_RELATIVE 3
#define R_RISCV_JUMP_SLOT 5
#define R_RISCV_IRELATIVE 58
#define R_RISCV_COPY 4
#define R_RISCV_TLS_DTPMOD64 7
#define R_RISCV_TLS_DTPREL64 9

// The no-op relocation is type zero on every architecture, and the absolute
// data relocations below are what object file tools match against.

// x86, complete: object file tools switch over every relocation type an
// input might carry, not just the ones this build emits.
#define R_386_16 20
#define R_386_32 1
#define R_386_32PLT 11
#define R_386_8 22
#define R_386_COPY 5
#define R_386_GLOB_DAT 6
#define R_386_GOT32 3
#define R_386_GOT32X 43
#define R_386_GOTOFF 9
#define R_386_GOTPC 10
#define R_386_IRELATIVE 42
#define R_386_JMP_SLOT 7
#define R_386_NONE 0
#define R_386_NUM 44
#define R_386_PC16 21
#define R_386_PC32 2
#define R_386_PC8 23
#define R_386_PLT32 4
#define R_386_RELATIVE 8
#define R_386_SIZE32 38
#define R_386_TLS_DESC 41
#define R_386_TLS_DESC_CALL 40
#define R_386_TLS_DTPMOD32 35
#define R_386_TLS_DTPOFF32 36
#define R_386_TLS_GD 18
#define R_386_TLS_GD_32 24
#define R_386_TLS_GD_CALL 26
#define R_386_TLS_GD_POP 27
#define R_386_TLS_GD_PUSH 25
#define R_386_TLS_GOTDESC 39
#define R_386_TLS_GOTIE 16
#define R_386_TLS_IE 15
#define R_386_TLS_IE_32 33
#define R_386_TLS_LDM 19
#define R_386_TLS_LDM_32 28
#define R_386_TLS_LDM_CALL 30
#define R_386_TLS_LDM_POP 31
#define R_386_TLS_LDM_PUSH 29
#define R_386_TLS_LDO_32 32
#define R_386_TLS_LE 17
#define R_386_TLS_LE_32 34
#define R_386_TLS_TPOFF 14
#define R_386_TLS_TPOFF32 37

#define R_X86_64_16 12
#define R_X86_64_32 10
#define R_X86_64_32S 11
#define R_X86_64_64 1
#define R_X86_64_8 14
#define R_X86_64_COPY 5
#define R_X86_64_DTPMOD64 16
#define R_X86_64_DTPOFF32 21
#define R_X86_64_DTPOFF64 17
#define R_X86_64_GLOB_DAT 6
#define R_X86_64_GOT32 3
#define R_X86_64_GOT64 27
#define R_X86_64_GOTOFF64 25
#define R_X86_64_GOTPC32 26
#define R_X86_64_GOTPC32_TLSDESC 34
#define R_X86_64_GOTPC64 29
#define R_X86_64_GOTPCREL 9
#define R_X86_64_GOTPCREL64 28
#define R_X86_64_GOTPCRELX 41
#define R_X86_64_GOTPLT64 30
#define R_X86_64_GOTTPOFF 22
#define R_X86_64_IRELATIVE 37
#define R_X86_64_JUMP_SLOT 7
#define R_X86_64_NONE 0
#define R_X86_64_NUM 43
#define R_X86_64_PC16 13
#define R_X86_64_PC32 2
#define R_X86_64_PC64 24
#define R_X86_64_PC8 15
#define R_X86_64_PLT32 4
#define R_X86_64_PLTOFF64 31
#define R_X86_64_RELATIVE 8
#define R_X86_64_RELATIVE64 38
#define R_X86_64_REX_GOTPCRELX 42
#define R_X86_64_SIZE32 32
#define R_X86_64_SIZE64 33
#define R_X86_64_TLSDESC 36
#define R_X86_64_TLSDESC_CALL 35
#define R_X86_64_TLSGD 19
#define R_X86_64_TLSLD 20
#define R_X86_64_TPOFF32 23
#define R_X86_64_TPOFF64 18

#define R_AARCH64_NONE 0

#define R_ARM_NONE 0

#define R_MIPS_32 2
#define R_MIPS_64 18

#define R_PPC_ADDR32 1
#define R_PPC64_ADDR64 38

#define R_390_64 22

#define R_SH_DIR32 1

#define R_SPARC_64 32

#define R_LARCH_32 1
#define R_LARCH_64 2
#define R_LARCH_MARK_LA 20
#define R_LARCH_SOP_PUSH_PLT_PCREL 29

// Additional relocation and symbol types that object file tools match
// against. R_ARM_THM_CALL is the current name for the value glibc still
// spells R_ARM_THM_PC22.
#define R_ARM_CALL 28
#define R_ARM_JUMP24 29
#define R_ARM_MOVT_ABS 44
#define R_ARM_MOVW_ABS_NC 43
#define R_ARM_PC24 1
#define R_ARM_REL32 3
#define R_ARM_THM_CALL 10
#define R_ARM_THM_JUMP19 51
#define R_ARM_THM_JUMP24 30
#define R_ARM_THM_MOVT_ABS 48
#define R_ARM_THM_MOVW_ABS_NC 47
#define R_ARM_THM_PC22 10

#define R_AARCH64_CALL26 283

#define R_MIPS_26 4
#define R_MIPS_LO16 6

#define R_RISCV_SUB32 39

#define R_LARCH_ALIGN 102
#define R_LARCH_RELAX 100
#define R_LARCH_SUB32 55

#define STT_SPARC_REGISTER 13

// The thirty two bit absolute data relocation of each remaining
// architecture, which is what a tool rewriting debug information matches
// against to find a value it has to adjust.
#define R_390_32 4
#define R_68K_32 1
#define R_ALPHA_REFLONG 1
#define R_IA64_SECREL32LSB 0x65
#define R_PARISC_DIR32 1
#define R_PPC_UADDR32 24
#define R_SPARC_32 3
#define R_SPARC_UA32 23

// The section a MIPS object keeps its debug information in.
#define SHT_MIPS_DWARF 0x7000001e

#endif // LLVM_LIBC_MACROS_ELF_MACROS_H
