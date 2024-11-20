//===-- xray_trampoline_riscv32.s ----------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This implements the riscv32-specific assembler for the trampolines.
//
//===----------------------------------------------------------------------===//

.macro SAVE_ARG_REGISTERS
	// Push argument registers to stack
	addi	sp, sp, -100
	.cfi_def_cfa_offset 100
	sw	ra, 96(sp)
	.cfi_offset ra, -4
	sw	a7, 92(sp)
	sw	a6, 88(sp)
	sw	a5, 84(sp)
	sw	a4, 80(sp)
	sw	a3, 76(sp)
	sw	a2, 72(sp)
	sw	a1, 68(sp)
	sw	a0, 64(sp)
	fsd	fa7, 56(sp)
	fsd	fa6, 48(sp)
	fsd	fa5, 40(sp)
	fsd	fa4, 32(sp)
	fsd	fa3, 24(sp)
	fsd	fa2, 16(sp)
	fsd	fa1, 8(sp)
	fsd	fa0, 0(sp)
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
	lw	a0, 64(sp)
	lw	a1, 68(sp)
	lw	a2, 72(sp)
	lw	a3, 76(sp)
	lw	a4, 80(sp)
	lw	a5, 84(sp)
	lw	a6, 88(sp)
	lw	a7, 92(sp)
	lw	ra, 96(sp)
	addi	sp, sp, 100
.endm

.macro SAVE_RET_REGISTERS
	// Push return registers to stack
	addi	sp, sp, -28
	.cfi_def_cfa_offset 28
	sw	ra, 24(sp)
	.cfi_offset ra, -4
	sw	a1, 20(sp)
	sw	a0, 16(sp)
	fsd	fa1, 8(sp)
	fsd	fa0, 0(sp)
.endm

.macro RESTORE_RET_REGISTERS
	// Restore return registers
	fld	fa0, 0(sp)
	fld	fa1, 8(sp)
	lw	a0, 16(sp)
	lw	a1, 20(sp)
	lw	ra, 24(sp)
	addi	sp, sp, 28
.endm

#include "xray_trampoline_riscv_common.S"
