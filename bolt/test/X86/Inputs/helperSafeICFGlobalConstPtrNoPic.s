# clang++ helper.cpp -c -o
# #define MY_CONST const
# int FooVar = 1;
# int BarVar = 2;
	.text
	.file	"helper.cpp"
	.type	FooVar,@object                  # @FooVar
	.data
	.globl	FooVar
	.p2align	2, 0x0
FooVar:
	.long	1                               # 0x1
	.size	FooVar, 4

	.type	BarVar,@object                  # @BarVar
	.globl	BarVar
	.p2align	2, 0x0
BarVar:
	.long	2                               # 0x2
	.size	BarVar, 4

	.ident	"clang version 20.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
