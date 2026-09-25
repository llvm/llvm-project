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
//   r31 = return address back to the sled (set by the call)
//
// The sled wraps the call with allocframe(#8)/deallocframe: this preserves the
// caller's original r31:30 (LR:FP) and provides an 8-byte slot in which the sled
// spills and reloads r7 (the only register the sled itself clobbers).

// 112-byte, 8-byte-aligned save area:
//   sp+#0:  r1:0    sp+#8:  r3:2    sp+#16: r5:4    sp+#24: r7:6
//   sp+#32: r9:8    sp+#40: r11:10  sp+#48: r13:12  sp+#56: r15:14
//   sp+#64: r28     sp+#68: r31     sp+#72: p3:0    sp+#76: usr
//   sp+#80: c1:0 (SA0,LC0)   sp+#88: c3:2 (SA1,LC1)   sp+#96: c7:6 (M0,M1)
//   (sp+#104: pad)
//
// The trampoline must preserve every caller-saved register, not just the
// argument registers: an XRay sled is inserted post-register-allocation and is
// treated by the compiler as clobbering nothing, but a function-exit sled can
// land in the middle of a loop (a conditional early return), where loop-carried
// values live across it.  The C++ handler freely clobbers the whole caller-saved
// set: GPRs r0-r15 and r28, predicates p0-p3, the hardware-loop registers
// SA0/LC0/SA1/LC1, the modifier registers M0/M1, and USR.  In particular a
// hardware loop (loop0/loop1) with an exit sled in its body needs SA0/LC0 (and
// SA1/LC1 if nested) preserved or the loop's trip count is corrupted.  r16-r27
// are callee-saved (preserved by the handler) and r29/r30 are managed by the
// sled's allocframe/deallocframe.  (The sled spills/reloads r7 itself, since it
// clobbers r7 with the funcid before this trampoline runs.)  GPRs are spilled as
// pairs with memd, two memory ops per packet; r0/r1 are reused as scratch for
// the control registers once their originals are on the stack.

.macro SAVE_REGISTERS
	// The paired memd reads the pre-packet r0/r1, so moving the predicates into
	// r0 in this same packet is safe (r0's original value is already stored).
	{
		memd(sp+#-112) = r1:0
		memd(sp+#-104) = r3:2
		sp = add(sp, #-112)
		r0 = p3:0
	}
	{
		memd(sp+#16) = r5:4
		memd(sp+#24) = r7:6
	}
	{
		memd(sp+#32) = r9:8
		memd(sp+#40) = r11:10
	}
	{
		memd(sp+#48) = r13:12
		memd(sp+#56) = r15:14
	}
	{
		memw(sp+#64) = r28
		memw(sp+#68) = r31
	}
	// Predicates (computed above) and the control registers.  r0/r1 are already
	// saved, so they are free to use as scratch for the control-register reads.
	// Each control-register transfer must be alone in its packet (slot 3 only).
	memw(sp+#72) = r0
	r1:0 = c1:0
	memd(sp+#80) = r1:0
	r1:0 = c3:2
	memd(sp+#88) = r1:0
	r1:0 = c7:6
	memd(sp+#96) = r1:0
	r0 = usr
	memw(sp+#76) = r0
.endm

.macro RESTORE_REGISTERS
	// Restore the control registers first (each transfer alone, slot 3 only),
	// using r0/r1 as scratch before the real r0/r1 are reloaded below; then the
	// predicates and GPRs.  All loads use the pre-update sp; sp is bumped last.
	r0 = memw(sp+#76)
	usr = r0
	r1:0 = memd(sp+#96)
	c7:6 = r1:0
	r1:0 = memd(sp+#88)
	c3:2 = r1:0
	r1:0 = memd(sp+#80)
	c1:0 = r1:0
	{
		r0 = memw(sp+#72)
		r31 = memw(sp+#68)
	}
	{
		p3:0 = r0
		r1:0 = memd(sp+#0)
		r3:2 = memd(sp+#8)
	}
	{
		r5:4 = memd(sp+#16)
		r7:6 = memd(sp+#24)
	}
	{
		r9:8 = memd(sp+#32)
		r11:10 = memd(sp+#40)
	}
	{
		r13:12 = memd(sp+#48)
		r15:14 = memd(sp+#56)
	}
	{
		r28 = memw(sp+#64)
		sp = add(sp, #112)
	}
.endm

.macro CALL_PATCHED_FUNC entry_type
	// Load the address of the global handler function pointer,
	// then dereference it to get the actual handler.
	r8 = add(pc,##ASM_SYMBOL(_ZN6__xray19XRayPatchedFunctionE)@PCREL)
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
	r8 = add(pc,##ASM_SYMBOL(_ZN6__xray22XRayPatchedCustomEventE)@PCREL)
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
	r8 = add(pc,##ASM_SYMBOL(_ZN6__xray21XRayPatchedTypedEventE)@PCREL)
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
