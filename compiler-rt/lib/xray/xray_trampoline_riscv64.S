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

.macro SAVE_ARG_REGISTERS
	// Push return registers to stack
	addi	sp, sp, -136
	.cfi_def_cfa_offset 136
        sd	ra, 128(sp)
        .cfi_offset ra, -8
        sd	a7, 120(sp)
        sd	a6, 112(sp)
        sd	a5, 104(sp)
        sd	a4, 96(sp)
        sd	a3, 88(sp)
        sd	a2, 80(sp)
        sd	a1, 72(sp)
        sd	a0, 64(sp)
        fsd	fa7, 56(sp)
        fsd	fa6, 48(sp)
        fsd	fa5, 40(sp)
        fsd	fa4, 32(sp)
        fsd	fa3, 24(sp)
        fsd	fa2, 16(sp)
        fsd	fa1, 8(sp)
        fsd	fa0, 0(sp)
.endm

.macro SAVE_RET_REGISTERS
	// Push return registers to stack
        addi    sp, sp, -40
        .cfi_def_cfa_offset 40
        sd      ra, 32(sp)
        .cfi_offset ra, -8
        sd      a1, 24(sp)
        sd      a0, 16(sp)
        fsd     fa1, 8(sp)
        fsd     fa0, 0(sp)
.endm

.macro RESTORE_RET_REGISTERS
	// Restore return registers
        fld     fa0, 0(sp)
        fld     fa1, 8(sp)
        ld      a0, 16(sp)
        ld      a1, 24(sp)
        ld      ra, 32(sp)
        addi    sp, sp, 40
.endm

.macro RESTORE_ARG_REGISTERS
        // Restore argument registers
	fld	fa0, 0(sp)
	fld	fa1, 8(sp)
	fld	fa2, 16(sp)
	fld	fa3, 24(sp)
	fld	fa4, 32(sp)
	fld	fa5, 40(sp)
	fld	fa6, 48(sp)
	fld	fa7, 56(sp)
	ld	a0, 64(sp)
	ld	a1, 72(sp)
	ld	a2, 80(sp)
	ld	a3, 88(sp)
	ld	a4, 96(sp)
	ld	a5, 104(sp)
	ld	a6, 112(sp)
	ld	a7, 120(sp)
	ld	ra, 128(sp)
	addi	sp, sp, 136
.endm

#include "xray_trampoline_riscv_common.S"
