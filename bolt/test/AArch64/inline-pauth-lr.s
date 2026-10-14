# This test checks that inlining functions with the pauth-lr variants of
# fused pointer-auth-and-return instructions is properly handled by BOLT.

# REQUIRES: system-linux

# RUN: %clang %cflags -march=armv9.5-a+pauth-lr -O0 %s -o %t.exe -Wl,-q
# RUN: llvm-bolt --inline-all --print-inline  --print-only=_Z3barP1A  \
# RUN: %t.exe -o %t.bolt  | FileCheck %s

# CHECK: BOLT-INFO: inlined 0 calls at 2 call sites in 2 iteration(s). Change in binary size: 16 bytes.
# CHECK: Binary Function "_Z3barP1A" after inlining {
# CHECK-NOT:  bl	_Z3fooP1A
# CHECK:      paciasppc
# CHECK-NEXT: ldr	x8, [x0]
# CHECK-NEXT: ldr	w0, [x8]
# CHECK-NEXT: autiasppcr x28
# CHECK-NEXT: paciasppc
# CHECK-NEXT: ldr	x7, [x0]
# CHECK-NEXT: ldr	w0, [x7]
# CHECK-NEXT: autiasppc _Z3bazP1A

	.text
	.globl	_Z3fooP1A
	.type	_Z3fooP1A,@function
_Z3fooP1A:
    paciasppc
	ldr	x8, [x0]
	ldr	w0, [x8]
	retaasppcr x28
	.size	_Z3fooP1A, .-_Z3fooP1A

	.text
	.globl	_Z3bazP1A
	.type	_Z3bazP1A,@function
_Z3bazP1A:
0:
    paciasppc
	ldr	x7, [x0]
	ldr	w0, [x7]
	retaasppc 0b
	.size	_Z3bazP1A, .-_Z3bazP1A

	.globl	_Z3barP1A
	.type	_Z3barP1A,@function
_Z3barP1A:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	bl	_Z3fooP1A
	bl	_Z3bazP1A
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
