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

#include "../sanitizer_common/sanitizer_asm.h"

.macro SAVE_ARG_REGISTERS
	// Push argument registers to stack
	addi	sp, sp, -112
        CFI_DEF_CFA_OFFSET(112)
	sw	ra, 108(sp)
	sw	a7, 104(sp)
	sw	a6, 100(sp)
	sw	a5, 96(sp)
	sw	a4, 92(sp)
	sw	a3, 88(sp)
	sw	a2, 84(sp)
	sw	a1, 80(sp)
	sw	a0, 76(sp)
	fsd	fa7, 64(sp)
	fsd	fa6, 56(sp)
	fsd	fa5, 48(sp)
	fsd	fa4, 40(sp)
	fsd	fa3, 32(sp)
	fsd	fa2, 24(sp)
	fsd	fa1, 16(sp)
	fsd	fa0, 8(sp)
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
	lw	a0, 76(sp)
	lw	a1, 80(sp)
	lw	a2, 84(sp)
	lw	a3, 88(sp)
	lw	a4, 92(sp)
	lw	a5, 96(sp)
	lw	a6, 100(sp)
	lw	a7, 104(sp)
	lw	ra, 108(sp)
	addi	sp, sp, 112
        CFI_DEF_CFA_OFFSET(0)
.endm

.macro SAVE_RET_REGISTERS
	// Push return registers to stack
	addi	sp, sp, -32
        CFI_DEF_CFA_OFFSET(32)
	sw	ra, 28(sp)
	sw	a1, 24(sp)
	sw	a0, 20(sp)
	fsd	fa1, 8(sp)
	fsd	fa0, 0(sp)
.endm

.macro RESTORE_RET_REGISTERS
	// Restore return registers
	fld	fa0, 0(sp)
	fld	fa1, 8(sp)
	lw	a0, 20(sp)
	lw	a1, 24(sp)
	lw	ra, 28(sp)
	addi	sp, sp, 32
        CFI_DEF_CFA_OFFSET(0)
.endm

.macro LOAD_XLEN, rd, src
        lw      \rd, \src
.endm

#include "xray_trampoline_riscv_common.S"
