//===-- xray_trampoline_hexagon.s -------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This implements the hexagon-specific assembler for the trampolines.
//
//===----------------------------------------------------------------------===//

#include "../builtins/assembly.h"
#include "../sanitizer_common/sanitizer_asm.h"

// The patched sled sets:
//   r7 = function ID
//   r6 = trampoline address (used by callr, then dead)
//   r31 = return address back to sled (set by callr)
//
// The sled wraps the callr with allocframe(#0)/deallocframe to preserve
// the caller's original r31:30 across the trampoline call.

.macro SAVE_REGISTERS
	// Allocate 32 bytes on the stack:
	//   sp+#0:  r0     (parameter / return value)
	//   sp+#4:  r1     (parameter / return value)
	//   sp+#8:  r2     (parameter)
	//   sp+#12: r3     (parameter)
	//   sp+#16: r4     (parameter)
	//   sp+#20: r5     (parameter)
	//   sp+#24: r31    (return address back to sled's deallocframe)
	//   sp+#28: (padding for 8-byte alignment)
	sp = add(sp, #-32)
	memw(sp+#0) = r0
	memw(sp+#4) = r1
	memw(sp+#8) = r2
	memw(sp+#12) = r3
	memw(sp+#16) = r4
	memw(sp+#20) = r5
	memw(sp+#24) = r31
.endm

.macro RESTORE_REGISTERS
	r0 = memw(sp+#0)
	r1 = memw(sp+#4)
	r2 = memw(sp+#8)
	r3 = memw(sp+#12)
	r4 = memw(sp+#16)
	r5 = memw(sp+#20)
	r31 = memw(sp+#24)
	sp = add(sp, #32)
.endm

.macro CALL_PATCHED_FUNC entry_type
	// Load the address of the global handler function pointer,
	// then dereference it to get the actual handler.
	r8 = ##ASM_SYMBOL(_ZN6__xray19XRayPatchedFunctionE)
	r8 = memw(r8+#0)
	// Skip if handler is not registered (null).
	{
		p0 = cmp.eq(r8, #0)
		if (p0.new) jump:nt 0f
	}
	// Set up arguments for the handler:
	//   r0 = FuncId (was placed in r7 by the patched sled)
	//   r1 = entry type (ENTRY=0, EXIT=1, TAIL=2)
	{
		r0 = r7
		r1 = \entry_type
	}
	callr r8
0:
.endm

	.text
	.globl ASM_SYMBOL(__xray_FunctionEntry)
	ASM_HIDDEN(__xray_FunctionEntry)
	ASM_TYPE_FUNCTION(__xray_FunctionEntry)
ASM_SYMBOL(__xray_FunctionEntry):
	CFI_STARTPROC
	SAVE_REGISTERS

	CALL_PATCHED_FUNC #0  // XRayEntryType::ENTRY

	RESTORE_REGISTERS
	jumpr r31
	ASM_SIZE(__xray_FunctionEntry)
	CFI_ENDPROC


	.globl ASM_SYMBOL(__xray_FunctionExit)
	ASM_HIDDEN(__xray_FunctionExit)
	ASM_TYPE_FUNCTION(__xray_FunctionExit)
ASM_SYMBOL(__xray_FunctionExit):
	CFI_STARTPROC
	SAVE_REGISTERS

	CALL_PATCHED_FUNC #1  // XRayEntryType::EXIT

	RESTORE_REGISTERS
	jumpr r31
	ASM_SIZE(__xray_FunctionExit)
	CFI_ENDPROC


	.globl ASM_SYMBOL(__xray_FunctionTailExit)
	ASM_HIDDEN(__xray_FunctionTailExit)
	ASM_TYPE_FUNCTION(__xray_FunctionTailExit)
ASM_SYMBOL(__xray_FunctionTailExit):
	CFI_STARTPROC
	SAVE_REGISTERS

	CALL_PATCHED_FUNC #2  // XRayEntryType::TAIL

	RESTORE_REGISTERS
	jumpr r31
	ASM_SIZE(__xray_FunctionTailExit)
	CFI_ENDPROC

// __xray_*Event have default visibility so that they can be referenced by user
// DSOs that do not link against the runtime.

	.globl ASM_SYMBOL(__xray_CustomEvent)
	ASM_TYPE_FUNCTION(__xray_CustomEvent)
ASM_SYMBOL(__xray_CustomEvent):
	CFI_STARTPROC
	SAVE_REGISTERS

	// Arguments (r0=ptr, r1=size) are already set by the inline sled.
	// Load the handler address.
	r8 = ##ASM_SYMBOL(_ZN6__xray22XRayPatchedCustomEventE)
	r8 = memw(r8+#0)
	{
		p0 = cmp.eq(r8, #0)
		if (p0.new) jump:nt 0f
	}
	callr r8
0:
	RESTORE_REGISTERS
	jumpr r31
	ASM_SIZE(__xray_CustomEvent)
	CFI_ENDPROC


	.globl ASM_SYMBOL(__xray_TypedEvent)
	ASM_TYPE_FUNCTION(__xray_TypedEvent)
ASM_SYMBOL(__xray_TypedEvent):
	CFI_STARTPROC
	SAVE_REGISTERS

	// Arguments (r0=type, r1=ptr, r2=size) are already set by the inline sled.
	// Load the handler address.
	r8 = ##ASM_SYMBOL(_ZN6__xray21XRayPatchedTypedEventE)
	r8 = memw(r8+#0)
	{
		p0 = cmp.eq(r8, #0)
		if (p0.new) jump:nt 0f
	}
	callr r8
0:
	RESTORE_REGISTERS
	jumpr r31
	ASM_SIZE(__xray_TypedEvent)
	CFI_ENDPROC

NO_EXEC_STACK_DIRECTIVE
