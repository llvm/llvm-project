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
	.p2align	4, 0x90
	.type	destructor,@function
destructor:
.Ldestructor$local:

	leaq	.L.str.d(%rip), %rdi
	jmp	puts@PLT

	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
.Lmain$local:

	pushq	%rax
	leaq	.L.str(%rip), %rdi
	callq	puts@PLT
	xorl	%eax, %eax
	popq	%rcx
	retq

	.p2align	4, 0x90
	.type	constructor.65535,@function
constructor.65535:

	pushq	%rax
	leaq	.L.str.65535(%rip), %rdi
	callq	puts@PLT
	leaq	.Ldestructor$local(%rip), %rdi
	leaq	__dso_handle(%rip), %rdx
	xorl	%esi, %esi
	popq	%rax
	jmp	__cxa_atexit@PLT

	.p2align	4, 0x90
	.type	constructor.200,@function
constructor.200:

	leaq	.L.str.200(%rip), %rdi
	jmp	puts@PLT

	.p2align	4, 0x90
	.type	constructor.100,@function
constructor.100:

	leaq	.L.str.100(%rip), %rdi
	jmp	puts@PLT

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

	.type	.L.str.d,@object
.L.str.d:
	.asciz	"destructor"
	.size	.L.str.d, 11

	.section	.init_array.100,"aw",@init_array
	.p2align	3
	.quad	constructor.100
	.section	.init_array.200,"aw",@init_array
	.p2align	3
	.quad	constructor.200
	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	constructor.65535
