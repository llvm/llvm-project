//===-- xray_trampoline_riscv64.s ----------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This implements the riscv64-specific assembler for the trampolines.
//
//===----------------------------------------------------------------------===//

#include "../sanitizer_common/sanitizer_asm.h"

.macro SAVE_ARG_REGISTERS
	// Push return registers to stack
	addi	sp, sp, -144
        CFI_DEF_CFA_OFFSET(144)
        sd	ra, 136(sp)
        sd	a7, 128(sp)
        sd	a6, 120(sp)
        sd	a5, 112(sp)
        sd	a4, 104(sp)
        sd	a3, 96(sp)
        sd	a2, 88(sp)
        sd	a1, 80(sp)
        sd	a0, 72(sp)
        fsd	fa7, 64(sp)
        fsd	fa6, 56(sp)
        fsd	fa5, 48(sp)
        fsd	fa4, 40(sp)
        fsd	fa3, 32(sp)
        fsd	fa2, 24(sp)
        fsd	fa1, 16(sp)
        fsd	fa0, 8(sp)
.endm

.macro SAVE_RET_REGISTERS
	// Push return registers to stack
        addi    sp, sp, -48
        CFI_DEF_CFA_OFFSET(48)
        sd      ra, 40(sp)
        sd      a1, 32(sp)
        sd      a0, 24(sp)
        fsd     fa1, 16(sp)
        fsd     fa0, 8(sp)
.endm

.macro RESTORE_RET_REGISTERS
	// Restore return registers
        fld     fa0, 8(sp)
        fld     fa1, 16(sp)
        ld      a0, 24(sp)
        ld      a1, 32(sp)
        ld      ra, 40(sp)
        addi    sp, sp, 48
        CFI_DEF_CFA_OFFSET(0)
.endm

.macro RESTORE_ARG_REGISTERS
        // Restore argument registers
	fld	fa0, 8(sp)
	fld	fa1, 16(sp)
	fld	fa2, 24(sp)
	fld	fa3, 32(sp)
	fld	fa4, 40(sp)
	fld	fa5, 48(sp)
	fld	fa6, 56(sp)
	fld	fa7, 64(sp)
	ld	a0, 72(sp)
	ld	a1, 80(sp)
	ld	a2, 88(sp)
	ld	a3, 96(sp)
	ld	a4, 104(sp)
	ld	a5, 112(sp)
	ld	a6, 120(sp)
	ld	a7, 128(sp)
	ld	ra, 136(sp)
	addi	sp, sp, 144
        CFI_DEF_CFA_OFFSET(0)
.endm

.macro LOAD_XLEN, rd, src
        ld      \rd, \src
.endm

#include "xray_trampoline_riscv_common.S"
