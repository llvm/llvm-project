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
	.abiversion 2
	.globl	constructor.100
	.p2align	4
	.type	constructor.100,@function
constructor.100:
.Lfunc_begin0:
.Lfunc_gep0:
	addis 2, 12, .TOC.-.Lfunc_gep0@ha
	addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
	.localentry	constructor.100, .Lfunc_lep0-.Lfunc_gep0
	mflr 0
	stdu 1, -32(1)
	std 0, 48(1)
	addis 3, 2, .L.str@toc@ha
	addi 3, 3, .L.str@toc@l
	bl puts
	nop
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	constructor.100, .Lfunc_end0-.Lfunc_begin0

	.globl	constructor.200
	.p2align	4
	.type	constructor.200,@function
constructor.200:
.Lfunc_begin1:
.Lfunc_gep1:
	addis 2, 12, .TOC.-.Lfunc_gep1@ha
	addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
	.localentry	constructor.200, .Lfunc_lep1-.Lfunc_gep1
	mflr 0
	stdu 1, -32(1)
	std 0, 48(1)
	addis 3, 2, .L.str.1@toc@ha
	addi 3, 3, .L.str.1@toc@l
	bl puts
	nop
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end1:
	.size	constructor.200, .Lfunc_end1-.Lfunc_begin1

	.globl	constructor.65535
	.p2align	4
	.type	constructor.65535,@function
constructor.65535:
.Lfunc_begin2:
.Lfunc_gep2:
	addis 2, 12, .TOC.-.Lfunc_gep2@ha
	addi 2, 2, .TOC.-.Lfunc_gep2@l
.Lfunc_lep2:
	.localentry	constructor.65535, .Lfunc_lep2-.Lfunc_gep2
	mflr 0
	stdu 1, -32(1)
	std 0, 48(1)
	addis 3, 2, .L.str.2@toc@ha
	addi 3, 3, .L.str.2@toc@l
	bl puts
	nop
	addis 3, 2, destructor@toc@ha
	addi 3, 3, destructor@toc@l
	addis 5, 2, __dso_handle@toc@ha
	addi 5, 5, __dso_handle@toc@l
	li 4, 0
	bl __cxa_atexit
	nop
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end2:
	.size	constructor.65535, .Lfunc_end2-.Lfunc_begin2

	.globl	destructor
	.p2align	4
	.type	destructor,@function
destructor:
.Lfunc_begin3:
.Lfunc_gep3:
	addis 2, 12, .TOC.-.Lfunc_gep3@ha
	addi 2, 2, .TOC.-.Lfunc_gep3@l
.Lfunc_lep3:
	.localentry	destructor, .Lfunc_lep3-.Lfunc_gep3
	mflr 0
	stdu 1, -32(1)
	std 0, 48(1)
	addis 3, 2, .L.str.3@toc@ha
	addi 3, 3, .L.str.3@toc@l
	bl puts
	nop
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end3:
	.size	destructor, .Lfunc_end3-.Lfunc_begin3

	.globl	main
	.p2align	4
	.type	main,@function
main:
.Lfunc_begin4:
.Lfunc_gep4:
	addis 2, 12, .TOC.-.Lfunc_gep4@ha
	addi 2, 2, .TOC.-.Lfunc_gep4@l
.Lfunc_lep4:
	.localentry	main, .Lfunc_lep4-.Lfunc_gep4
	mflr 0
	stdu 1, -32(1)
	std 0, 48(1)
	addis 3, 2, .L.str.4@toc@ha
	addi 3, 3, .L.str.4@toc@l
	bl puts
	nop
	li 3, 0
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end4:
	.size	main, .Lfunc_end4-.Lfunc_begin4

	.hidden __dso_handle
	.type	.L.str,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"constructor 100"
	.size	.L.str, 16

	.type	.L.str.1,@object
.L.str.1:
	.asciz	"constructor 200"
	.size	.L.str.1, 16

	.type	.L.str.2,@object
.L.str.2:
	.asciz	"constructor 65535"
	.size	.L.str.2, 18

	.type	.L.str.3,@object
.L.str.3:
	.asciz	"destructor"
	.size	.L.str.3, 11

	.type	.L.str.4,@object
.L.str.4:
	.asciz	"main"
	.size	.L.str.4, 5

	.section	.init_array.100,"aw",@init_array
	.p2align	3
	.quad	constructor.100
	.section	.init_array.200,"aw",@init_array
	.p2align	3
	.quad	constructor.200
	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	constructor.65535
