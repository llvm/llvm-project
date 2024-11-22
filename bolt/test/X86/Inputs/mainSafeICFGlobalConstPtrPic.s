# clang++ main.cpp -c -o
# #define MY_CONST const
# extern int FooVar;
# extern int BarVar;
# [[clang::noinline]]
# MY_CONST int fooMul(int a, int b) {
#   return a * b;
# }
# [[clang::noinline]]
# MY_CONST int barMul(int a, int b) {
#   return a * b;
# }
# [[clang::noinline]]
# MY_CONST int fooAdd(int a, int b) {
#   return a + b;
# }
# [[clang::noinline]]
# MY_CONST int barAdd(int a, int b) {
#   return a + b;
# }
# [[clang::noinline]]
# MY_CONST int helper1(MY_CONST int (*func)(int, int), int a, int b) {
#   if (func == barAdd)
#     return 1;
#   return func(a, b) - 4;
# }
# [[clang::noinline]]
# MY_CONST int helper2(MY_CONST int (*func)(int, int), MY_CONST int (*func2)(int, int), int a, int b) {
#   if (func == func2)
#     return 2;
#   return func(a, b) + func2(a, b);
# }
# MY_CONST static int (*MY_CONST funcGlobalBarAdd)(int, int) = barAdd;
# MY_CONST int (*MY_CONST funcGlobalBarMul)(int, int) = barMul;
# int main(int argc, char **argv) {
#   int temp = helper1(funcGlobalBarAdd, FooVar, BarVar) +
#              helper2(fooMul, funcGlobalBarMul, FooVar, BarVar) +
#              fooAdd(FooVar, BarVar);
#   MY_PRINTF("val: %d", temp);
#   return temp;
# }
# Manually modified to remove "extra" assembly.
	.text
	.file	"main.cpp"
	.globl	_Z6fooMulii                     # -- Begin function _Z6fooMulii
	.p2align	4, 0x90
	.type	_Z6fooMulii,@function
_Z6fooMulii:                            # @_Z6fooMulii
	.cfi_startproc
	imull	-8(%rbp), %eax
	retq
.Lfunc_end0:
	.size	_Z6fooMulii, .Lfunc_end0-_Z6fooMulii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6barMulii                     # -- Begin function _Z6barMulii
	.p2align	4, 0x90
	.type	_Z6barMulii,@function
_Z6barMulii:                            # @_Z6barMulii
	.cfi_startproc
	imull	-8(%rbp), %eax
	retq
.Lfunc_end1:
	.size	_Z6barMulii, .Lfunc_end1-_Z6barMulii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6fooAddii                     # -- Begin function _Z6fooAddii
	.p2align	4, 0x90
	.type	_Z6fooAddii,@function
_Z6fooAddii:                            # @_Z6fooAddii
	.cfi_startproc
	addl	-8(%rbp), %eax
	retq
.Lfunc_end2:
	.size	_Z6fooAddii, .Lfunc_end2-_Z6fooAddii
	.cfi_endproc
                                        # -- End function
	.globl	_Z6barAddii                     # -- Begin function _Z6barAddii
	.p2align	4, 0x90
	.type	_Z6barAddii,@function
_Z6barAddii:                            # @_Z6barAddii
	.cfi_startproc
	addl	-8(%rbp), %eax
	retq
.Lfunc_end3:
	.size	_Z6barAddii, .Lfunc_end3-_Z6barAddii
	.cfi_endproc
                                        # -- End function
	.globl	_Z7helper1PFKiiiEii             # -- Begin function _Z7helper1PFKiiiEii
	.p2align	4, 0x90
	.type	_Z7helper1PFKiiiEii,@function
_Z7helper1PFKiiiEii:                    # @_Z7helper1PFKiiiEii
	.cfi_startproc
	leaq	_Z6barAddii(%rip), %rax
	cmpq	%rax, -16(%rbp)
	retq
.Lfunc_end4:
	.size	_Z7helper1PFKiiiEii, .Lfunc_end4-_Z7helper1PFKiiiEii
	.cfi_endproc
                                        # -- End function
	.globl	_Z7helper2PFKiiiES1_ii          # -- Begin function _Z7helper2PFKiiiES1_ii
	.p2align	4, 0x90
	.type	_Z7helper2PFKiiiES1_ii,@function
_Z7helper2PFKiiiES1_ii:                 # @_Z7helper2PFKiiiES1_ii
	.cfi_startproc
  # Operates on registers.
	retq
.Lfunc_end5:
	.size	_Z7helper2PFKiiiES1_ii, .Lfunc_end5-_Z7helper2PFKiiiES1_ii
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
	movq	FooVar@GOTPCREL(%rip), %rax
	movq	BarVar@GOTPCREL(%rip), %rax
	leaq	_Z6barAddii(%rip), %rdi
	callq	_Z7helper1PFKiiiEii
	movq	FooVar@GOTPCREL(%rip), %rax
	movq	BarVar@GOTPCREL(%rip), %rax
	leaq	_Z6fooMulii(%rip), %rdi
	leaq	_Z6barMulii(%rip), %rsi
	callq	_Z7helper2PFKiiiES1_ii
	retq
.Lfunc_end6:
	.size	main, .Lfunc_end6-main
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 20.0.0git"
