# clang++ -O3 -c main.cpp -o main.o
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
#   static int (*funcGlobalBarAdd)(int, int) = barAdd;
#   int (*funcGlobalBarMul)(int, int) = barMul;
#   int temp = helper1(funcGlobalBarAdd, FooVar, BarVar) +
#              helper2(fooMul, funcGlobalBarMul, FooVar, BarVar) + fooSub(FooVar, BarVar) +
#              barSub(FooVar, BarVar) + fooAdd(FooVar, BarVar);
#   MY_PRINTF("val: %d", temp);
#   return temp;
# }
	.text
	.file	"main.cpp"
	.globl	_Z6fooSubii                     # -- Begin function _Z6fooSubii
	.p2align	4, 0x90
	.type	_Z6fooSubii,@function
_Z6fooSubii:                            # @_Z6fooSubii
	.cfi_startproc
	subl	%esi, %eax
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
	subl	%esi, %eax
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
	imull	%esi, %eax
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
	imull	%esi, %eax
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
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
	leal	(%rdi,%rsi), %eax
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
                                        # kill: def $esi killed $esi def $rsi
                                        # kill: def $edi killed $edi def $rdi
	leal	(%rdi,%rsi), %eax
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
	leaq	_Z6barAddii(%rip), %rcx
	cmpq	%rcx, %rdi
	retq
.LBB6_1:
	movl	$1, %eax
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
  # Operates on registers.
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
	movq	FooVar@GOTPCREL(%rip), %r14
	movq	BarVar@GOTPCREL(%rip), %r15
	leaq	_Z6barAddii(%rip), %rdi
	callq	_Z7helper1PFiiiEii
	leaq	_Z6fooMulii(%rip), %rdi
	leaq	_Z6barMulii(%rip), %rsi
	callq	_Z7helper2PFiiiES0_ii
	callq	_Z6fooSubii
	callq	_Z6barSubii
	callq	_Z6fooAddii
	retq
.Lfunc_end8:
	.size	main, .Lfunc_end8-main
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 20.0.0git"
