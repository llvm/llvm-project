# RUN: %clangxx %cflags -no-pie %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.exe.bolt
# RUN: %t.exe.bolt

# REQUIRES: system-linux

## Test that BOLT properly handles LPStart when LPStartEncoding is different
## from DW_EH_PE_omit.

# The test case compiled with -O1 from:
#
# int main() {
#   try {
#     throw 42;
#   } catch (...) {
#     return 0;
#   }
#   return 1;
# }
#
# The exception table was modified with udata4 LPStartEncoding and sdata4
# CallSiteEncoding.

	.text
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception0
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$4, %edi
	callq	__cxa_allocate_exception
	movl	$42, (%rax)
.Ltmp0:
	movl	$_ZTIi, %esi
	movq	%rax, %rdi
	xorl	%edx, %edx
	callq	__cxa_throw
.Ltmp1:
# %bb.1:
.LBB0_2:
.Ltmp2:
	movq	%rax, %rdi
	callq	__cxa_begin_catch
	callq	__cxa_end_catch
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table0:
.Lexception0:
	.byte	3                               # @LPStart Encoding = udata4
  .long 0
	.byte	3                               # @TType Encoding = udata4
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
	.long .Ltmp2                         #     jumps to .Ltmp2
	.byte	1                               #   On action: 1
	.long .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.long .Lfunc_end0-.Ltmp1             #   Call between .Ltmp1 and .Lfunc_end0
	.long	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase0:
	.p2align	2
                                        # -- End function
