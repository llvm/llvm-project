# clang++ -c main.cpp -o main.o
# extern int FooVar;
# extern int BarVar;
# [[clang::noinline]]
# int fooSub(int a, int b) {
#   return a - b;
# }
# [[clang::noinline]]
# int barSub(int a, int b) {
#   return a - b;
# }
# [[clang::noinline]]
# int fooMul(int a, int b) {
#   return a * b;
# }
# [[clang::noinline]]
# int barMul(int a, int b) {
#   return a * b;
# }
# [[clang::noinline]]
# int fooAdd(int a, int b) {
#   return a + b;
# }
# [[clang::noinline]]
# int barAdd(int a, int b) {
#   return a + b;
# }
# [[clang::noinline]]
# int helper1(int (*func)(int, int), int a, int b) {
#   if (func == barAdd)
#     return 1;
#   return func(a, b) - 4;
# }
# [[clang::noinline]]
# int helper2(int (*func)(int, int), int (*func2)(int, int), int a, int b) {
#   if (func == func2)
#     return 2;
#   return func(a, b) + func2(a, b);
# }
# int main(int argc, char **argv) {
#   int temp = helper1(barAdd, FooVar, BarVar) +
#              helper2(fooMul, barMul, FooVar, BarVar) + fooSub(FooVar, BarVar) +
#              barSub(FooVar, BarVar) + fooAdd(FooVar, BarVar);
#   return temp;
# }
	.text
	.file	"main.cpp"
	.globl	_Z6fooSubii                     # -- Begin function _Z6fooSubii
	.p2align	4, 0x90
	.type	_Z6fooSubii,@function
_Z6fooSubii:                            # @_Z6fooSubii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	subl	-8(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	_Z6fooSubii, .Lfunc_end0-_Z6fooSubii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6barSubii                     # -- Begin function _Z6barSubii
	.p2align	4, 0x90
	.type	_Z6barSubii,@function
_Z6barSubii:                            # @_Z6barSubii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	subl	-8(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	_Z6barSubii, .Lfunc_end1-_Z6barSubii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6fooMulii                     # -- Begin function _Z6fooMulii
	.p2align	4, 0x90
	.type	_Z6fooMulii,@function
_Z6fooMulii:                            # @_Z6fooMulii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	imull	-8(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	_Z6fooMulii, .Lfunc_end2-_Z6fooMulii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6barMulii                     # -- Begin function _Z6barMulii
	.p2align	4, 0x90
	.type	_Z6barMulii,@function
_Z6barMulii:                            # @_Z6barMulii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	imull	-8(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end3:
	.size	_Z6barMulii, .Lfunc_end3-_Z6barMulii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6fooAddii                     # -- Begin function _Z6fooAddii
	.p2align	4, 0x90
	.type	_Z6fooAddii,@function
_Z6fooAddii:                            # @_Z6fooAddii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	addl	-8(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end4:
	.size	_Z6fooAddii, .Lfunc_end4-_Z6fooAddii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6barAddii                     # -- Begin function _Z6barAddii
	.p2align	4, 0x90
	.type	_Z6barAddii,@function
_Z6barAddii:                            # @_Z6barAddii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	%edi, -4(%rbp)
	movl	%esi, -8(%rbp)
	movl	-4(%rbp), %eax
	addl	-8(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end5:
	.size	_Z6barAddii, .Lfunc_end5-_Z6barAddii
	.cfi_endproc
                                        # -- End function
	.globl	_Z7helper1PFiiiEii              # -- Begin function _Z7helper1PFiiiEii
	.p2align	4, 0x90
	.type	_Z7helper1PFiiiEii,@function
_Z7helper1PFiiiEii:                     # @_Z7helper1PFiiiEii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -16(%rbp)
	movl	%esi, -20(%rbp)
	movl	%edx, -24(%rbp)
	leaq	_Z6barAddii(%rip), %rax
	cmpq	%rax, -16(%rbp)
	jne	.LBB6_2
# %bb.1:
	movl	$1, -4(%rbp)
	jmp	.LBB6_3
.LBB6_2:
	movq	-16(%rbp), %rax
	movl	-20(%rbp), %edi
	movl	-24(%rbp), %esi
	callq	*%rax
	subl	$4, %eax
	movl	%eax, -4(%rbp)
.LBB6_3:
	movl	-4(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end6:
	.size	_Z7helper1PFiiiEii, .Lfunc_end6-_Z7helper1PFiiiEii
	.cfi_endproc
                                        # -- End function
	.globl	_Z7helper2PFiiiES0_ii           # -- Begin function _Z7helper2PFiiiES0_ii
	.p2align	4, 0x90
	.type	_Z7helper2PFiiiES0_ii,@function
_Z7helper2PFiiiES0_ii:                  # @_Z7helper2PFiiiES0_ii
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -16(%rbp)
	movq	%rsi, -24(%rbp)
	movl	%edx, -28(%rbp)
	movl	%ecx, -32(%rbp)
	movq	-16(%rbp), %rax
	cmpq	-24(%rbp), %rax
	jne	.LBB7_2
# %bb.1:
	movl	$2, -4(%rbp)
	jmp	.LBB7_3
.LBB7_2:
	movq	-16(%rbp), %rax
	movl	-28(%rbp), %edi
	movl	-32(%rbp), %esi
	callq	*%rax
	movl	%eax, -36(%rbp)                 # 4-byte Spill
	movq	-24(%rbp), %rax
	movl	-28(%rbp), %edi
	movl	-32(%rbp), %esi
	callq	*%rax
	movl	%eax, %ecx
	movl	-36(%rbp), %eax                 # 4-byte Reload
	addl	%ecx, %eax
	movl	%eax, -4(%rbp)
.LBB7_3:
	movl	-4(%rbp), %eax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end7:
	.size	_Z7helper2PFiiiES0_ii, .Lfunc_end7-_Z7helper2PFiiiES0_ii
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	FooVar@GOTPCREL(%rip), %rax
	movl	(%rax), %esi
	movq	BarVar@GOTPCREL(%rip), %rax
	movl	(%rax), %edx
	leaq	_Z6barAddii(%rip), %rdi
	callq	_Z7helper1PFiiiEii
	movl	%eax, -36(%rbp)                 # 4-byte Spill
	movq	FooVar@GOTPCREL(%rip), %rax
	movl	(%rax), %edx
	movq	BarVar@GOTPCREL(%rip), %rax
	movl	(%rax), %ecx
	leaq	_Z6fooMulii(%rip), %rdi
	leaq	_Z6barMulii(%rip), %rsi
	callq	_Z7helper2PFiiiES0_ii
	movl	%eax, %ecx
	movl	-36(%rbp), %eax                 # 4-byte Reload
	addl	%ecx, %eax
	movl	%eax, -32(%rbp)                 # 4-byte Spill
	movq	FooVar@GOTPCREL(%rip), %rax
	movl	(%rax), %edi
	movq	BarVar@GOTPCREL(%rip), %rax
	movl	(%rax), %esi
	callq	_Z6fooSubii
	movl	%eax, %ecx
	movl	-32(%rbp), %eax                 # 4-byte Reload
	addl	%ecx, %eax
	movl	%eax, -28(%rbp)                 # 4-byte Spill
	movq	FooVar@GOTPCREL(%rip), %rax
	movl	(%rax), %edi
	movq	BarVar@GOTPCREL(%rip), %rax
	movl	(%rax), %esi
	callq	_Z6barSubii
	movl	%eax, %ecx
	movl	-28(%rbp), %eax                 # 4-byte Reload
	addl	%ecx, %eax
	movl	%eax, -24(%rbp)                 # 4-byte Spill
	movq	FooVar@GOTPCREL(%rip), %rax
	movl	(%rax), %edi
	movq	BarVar@GOTPCREL(%rip), %rax
	movl	(%rax), %esi
	callq	_Z6fooAddii
	movl	%eax, %ecx
	movl	-24(%rbp), %eax                 # 4-byte Reload
	addl	%ecx, %eax
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end8:
	.size	main, .Lfunc_end8-main
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 20.0.0git"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z6fooSubii
	.addrsig_sym _Z6barSubii
	.addrsig_sym _Z6fooMulii
	.addrsig_sym _Z6barMulii
	.addrsig_sym _Z6fooAddii
	.addrsig_sym _Z6barAddii
	.addrsig_sym _Z7helper1PFiiiEii
	.addrsig_sym _Z7helper2PFiiiES0_ii
	.addrsig_sym FooVar
	.addrsig_sym BarVar
