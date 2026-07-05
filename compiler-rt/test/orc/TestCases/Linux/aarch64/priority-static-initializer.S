// Test that ELF static initializers with different constructor priorities work
// and are executed in the proper order.
//
// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t | FileCheck %s

// CHECK: constructor 100
// CHECK-NEXT: constructor 200
// CHECK-NEXT: constructor 65535
// CHECK-NEXT: main
// CHECK-NEXT: destructor

	.text
	.globl	destructor
	.p2align	2
	.type	destructor,@function
destructor:
.Ldestructor$local:

	adrp	x0, .L.str.2
	add	x0, x0, :lo12:.L.str.2
	b	puts

	.globl	main
	.p2align	2
	.type	main,@function
main:

	str	x30, [sp, #-16]!
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	bl	puts
	mov	w0, wzr
	ldr	x30, [sp], #16
	ret

	.p2align	2
	.type	constructor.65535,@function
constructor.65535:

	str	x30, [sp, #-16]!
	adrp	x0, .L.str.65535
	add	x0, x0, :lo12:.L.str.65535
	bl	puts
	adrp	x0, .Ldestructor$local
	add	x0, x0, :lo12:.Ldestructor$local
	adrp	x2, __dso_handle
	add	x2, x2, :lo12:__dso_handle
	mov	x1, xzr
	ldr	x30, [sp], #16
	b	__cxa_atexit

	.p2align	2
	.type	constructor.100,@function
constructor.100:

	str	x30, [sp, #-16]!
	adrp	x0, .L.str.100
	add	x0, x0, :lo12:.L.str.100
	bl	puts
	ldr	x30, [sp], #16
	ret

	.p2align	2
	.type	constructor.200,@function
constructor.200:

	str	x30, [sp, #-16]!
	adrp	x0, .L.str.200
	add	x0, x0, :lo12:.L.str.200
	bl	puts
	ldr	x30, [sp], #16
	ret

	.hidden	__dso_handle
	.type	.L.str,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"main"
	.size	.L.str, 5

	.type	.L.str.100,@object
.L.str.100:
	.asciz	"constructor 100"
	.size	.L.str.100, 16

	.type	.L.str.200,@object
.L.str.200:
	.asciz	"constructor 200"
	.size	.L.str.200, 16

	.type	.L.str.65535,@object
.L.str.65535:
	.asciz	"constructor 65535"
	.size	.L.str.65535, 18


	.type	.L.str.2,@object
.L.str.2:
	.asciz	"destructor"
	.size	.L.str.2, 11

	.section	.init_array.100,"aw",@init_array
	.p2align	3
	.xword	constructor.100
	.section	.init_array.200,"aw",@init_array
	.p2align	3
	.xword	constructor.200
	.section	.init_array,"aw",@init_array
	.p2align	3
	.xword	constructor.65535
