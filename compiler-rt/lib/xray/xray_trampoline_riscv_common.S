//===-- xray_trampoline_riscv_common.s --------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This implements the trampolines code shared between riscv32 and riscv64.
//
//===----------------------------------------------------------------------===//

#include "../builtins/assembly.h"

	.text
	.p2align 2
	.global ASM_SYMBOL(__xray_FunctionEntry)
	ASM_TYPE_FUNCTION(__xray_FunctionEntry)
ASM_SYMBOL(__xray_FunctionEntry):
	CFI_STARTPROC
        SAVE_ARG_REGISTERS

	// Load the handler function pointer into a2
	la	a2, ASM_SYMBOL(_ZN6__xray19XRayPatchedFunctionE)
        LOAD_XLEN   a2, 0(a2)

	// Handler address will be null if it is not set
	beq	a2, x0, 1f

	// If we reach here, we are tracing an event
	// a0 already contains function id
	// a1 = 0 means we are tracing an entry event
	li	a1, 0
	jalr	a2

1:
        RESTORE_ARG_REGISTERS
	jr	ra
	ASM_SIZE(__xray_FunctionEntry)
	CFI_ENDPROC

	.text
	.p2align 2
	.global ASM_SYMBOL(__xray_FunctionExit)
	ASM_TYPE_FUNCTION(__xray_FunctionExit)
ASM_SYMBOL(__xray_FunctionExit):
	CFI_STARTPROC
	SAVE_RET_REGISTERS

	// Load the handler function pointer into a2
	la	a2, ASM_SYMBOL(_ZN6__xray19XRayPatchedFunctionE)
        LOAD_XLEN   a2, 0(a2)

	// Handler address will be null if it is not set
	beq	a2, x0, 1f

	// If we reach here, we are tracing an event
	// a0 already contains function id
	// a1 = 1 means we are tracing an exit event
	li	a1, 1
	jalr	a2

1:
	RESTORE_RET_REGISTERS
        jr	ra
	ASM_SIZE(__xray_FunctionExit)
	CFI_ENDPROC

	.text
	.p2align 2
	.global ASM_SYMBOL(__xray_FunctionTailExit)
	ASM_TYPE_FUNCTION(__xray_FunctionTailExit)
ASM_SYMBOL(__xray_FunctionTailExit):
	CFI_STARTPROC
        SAVE_ARG_REGISTERS

	// Load the handler function pointer into a2
	la	a2, ASM_SYMBOL(_ZN6__xray19XRayPatchedFunctionE)
        LOAD_XLEN   a2, 0(a2)

	// Handler address will be null if it is not set
	beq	a2, x0, 1f

	// If we reach here, we are tracing an event
	// a0 already contains function id
	// a1 = 2 means we are tracing a tail exit event
	li	a1, 2
	jalr	a2

1:
        RESTORE_ARG_REGISTERS
	jr	ra
	ASM_SIZE(__xray_FunctionTailExit)
	CFI_ENDPROC
