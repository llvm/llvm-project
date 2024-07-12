# RUN: %{cxx} %{flags} %s %{link_flags} -no-pie -o %t.exe
# RUN: %{exec} %t.exe

# REQUIRES: linux && target={{x86_64-.+}}
# UNSUPPORTED: target={{.+-android.*}}
# UNSUPPORTED: no-exceptions

## Check that libc++abi works correctly when LPStart address is explicitly set
## to zero.

## This file is generated from the following C++ source code.
##
## ```
## int main() {
##   try {
##     throw 42;
##   } catch (...) {
##     return 0;
##   }
##   return 1;
## }
## ```
## The exception table is modified to use udata4 encoding for LPStart and
## sdata4 encoding for call sites.

	.text
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.globl	__gxx_personality_v0
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movl	$0, -4(%rbp)
	movl	$4, %edi
	callq	__cxa_allocate_exception@PLT
	movq	%rax, %rdi
	movl	$42, (%rdi)
.Ltmp0:
	movq	_ZTIi@GOTPCREL(%rip), %rsi
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	__cxa_throw@PLT
.Ltmp1:
	jmp	.LBB0_4
.LBB0_1:                                # %lpad
.Ltmp2:
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -16(%rbp)
	movl	%eax, -20(%rbp)
# %bb.2:                                # %catch
	movq	-16(%rbp), %rdi
	callq	__cxa_begin_catch@PLT
	movl	$0, -4(%rbp)
	callq	__cxa_end_catch@PLT
# %bb.3:                                # %return
	movl	-4(%rbp), %eax
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB0_4:                                # %unreachable
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc

	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	3                               # @LPStart Encoding = udata4
  .long 0
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	11                              # Call site Encoding = sdata4
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.long .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.long .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.long	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.long .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.long .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
  .long .Ltmp2
	.byte	1                               #   On action: 1
	.long .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.long .Lfunc_end0-.Ltmp1             #   Call between .Ltmp1 and .Lfunc_end0
	.long	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase0:
	.p2align	2, 0x0
                                        # -- End function
