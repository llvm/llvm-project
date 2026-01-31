# This test checks that for AArch64 binaries with BTI, we do not inline blocks with indirect tailcalls.
# Same as inline-bti.s, but checks the debug output, and therefore requires assertions.

# REQUIRES: system-linux, assertions

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -O0 %t.o -o %t.exe -Wl,-q -Wl,-z,force-bti
# RUN: llvm-bolt --inline-all %t.exe -o %t.bolt --debug 2>&1 | FileCheck %s

# For BTI, we should not inline foo.
# CHECK: BOLT-DEBUG: Skipping inlining block with tailcall in _Z3barP1A : .LBB01 to keep BTIs consistent.
# CHECK-NOT: BOLT-INFO: inlined {{[0-9]+}} calls at {{[0-9]+}} call sites in {{[0-9]+}} iteration(s). Change in binary size: {{[0-9]+}} bytes.

	.text
	.globl	_Z3fooP1A
	.type	_Z3fooP1A,@function
_Z3fooP1A:
	ldr	x8, [x0]
	ldr	w0, [x8]
	br x30
	.size	_Z3fooP1A, .-_Z3fooP1A

	.globl	_Z3barP1A
	.type	_Z3barP1A,@function
_Z3barP1A:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	bl	_Z3fooP1A
	mul	w0, w0, w0
	ldp	x29, x30, [sp], #16
	ret
	.size	_Z3barP1A, .-_Z3barP1A

	.globl	main
	.p2align	2
	.type	main,@function
main:
	mov	w0, wzr
	ret
	.size	main, .-main
