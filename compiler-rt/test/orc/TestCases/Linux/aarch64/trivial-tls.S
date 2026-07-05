// RUN: %clang -c -o %t %s
// RUN: %llvm_jitlink %t
//
// Test that basic ELF TLS work by adding together TLSs with values
// 0, 1, and -1, and returning the result (0 for success). This setup
// tests both zero-initialized (.tbss) and non-zero-initialized
// (.tdata) sections.

	.text
	.file	"tlstest.cpp"
	.globl	main
	.p2align	2
	.type	main,@function
main:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	adrp	x0, :tlsdesc:x
	ldr	x1, [x0, :tlsdesc_lo12:x]
	add	x0, x0, :tlsdesc_lo12:x
	.tlsdesccall x
	blr	x1
	mrs	x8, TPIDR_EL0
	ldr	w9, [x8, x0]
	adrp	x0, :tlsdesc:y
	ldr	x1, [x0, :tlsdesc_lo12:y]
	add	x0, x0, :tlsdesc_lo12:y
	.tlsdesccall y
	blr	x1
	ldr	w10, [x8, x0]
	add	w9, w10, w9
	adrp	x0, :tlsdesc:z
	ldr	x1, [x0, :tlsdesc_lo12:z]
	add	x0, x0, :tlsdesc_lo12:z
	.tlsdesccall z
	blr	x1
	ldr	w8, [x8, x0]
	add	w0, w9, w8
	ldp	x29, x30, [sp], #16
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main

	.type	x,@object
	.section	.tdata,"awT",@progbits
	.globl	x
	.p2align	2
x:
	.word	4294967295
	.size	x, 4

	.type	y,@object
	.section	.tbss,"awT",@nobits
	.globl	y
	.p2align	2
y:
	.word	0
	.size	y, 4

	.type	z,@object
	.section	.tdata,"awT",@progbits
	.globl	z
	.p2align	2
z:
	.word	1
	.size	z, 4
  