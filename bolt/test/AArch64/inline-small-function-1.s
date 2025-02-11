## This test checks that inline is properly handled by BOLT on aarch64.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -O0 %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt --inline-small-functions --print-inline  --print-only=_Z3barP1A  \
# RUN: %t.exe -o %t.bolt  | FileCheck %s

# CHECK: BOLT-INFO: inlined 0 calls at 1 call sites in 2 iteration(s). Change in binary size: 4 bytes.
# CHECK: Binary Function "_Z3barP1A" after inlining {
# CHECK-NOT: bl	_Z3fooP1A
# CHECK: ldr	x8, [x0]
# CHECK-NEXT: ldr	w0, [x8]
 
	.text
	.globl	_Z3fooP1A                      
	.type	_Z3fooP1A,@function
_Z3fooP1A:                              
	ldr	x8, [x0]
	ldr	w0, [x8]
	ret
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
