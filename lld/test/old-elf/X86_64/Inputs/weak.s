	.file	"weak.s"
	.text
	.p2align 4,,15
	.globl	test
	.type	test, @function
test:
	ret
	.size	test, .-test
	.weak	myfn2
	.data
	.align 8
	.type	myfn2, @object
	.size	myfn2, 8
myfn2:
	.quad	test
	.weak	myfn1
	.align 8
	.type	myfn1, @object
	.size	myfn1, 8
myfn1:
	.quad	test
