# Assembly generated from building the following C++ code with the following
# command using trunk clang. Then, basic block at .LBB1_7 was moved before the
# landing pad.
#
# clang --target=x86_64-linux -O2 -fPIC -fno-inline exceptions-failed-split.cpp
#
#		#include <stdio.h>
#		#include <stdint.h>
#
#		void bar(int a) {
#			if (a > 2 && a % 2)
#				throw new int();
#		}
#
#		uint64_t throw_test(int argc, char **argv) {
#			uint64_t rv = 0;
#
#			if (argc == 99)
#				return 0;
#
#			uint64_t limit = (argc >= 2 ? 10 : 5000);
#			for (uint64_t i = 0; i < limit; ++i) {
#				rv += i;
#				try  {
#					bar(argc);
#				} catch (...) {
#				}
#			}
#
#			if (argc == 5)
#				return 0;
#
#			if (argc == 7)
#				return 0;
#
#			if (argc >= 103 && argc <= 203)
#				return 0;
#
#			if (*argv == 0)
#				return 0;
#
#			if (argc >= 13 && argc <= 23)
#				return 0;
#
#			return rv;
#		}
#
#		int main(int argc, char **argv) {
#			return !throw_test(argc, argv);
#		}

	.text
	.file	"exceptions-failed-split.cpp"
	.globl	_Z3bari                         # -- Begin function _Z3bari
	.p2align	4, 0x90
	.type	_Z3bari,@function
_Z3bari:                                # @_Z3bari
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:                                # %entry
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	pushq	%rax
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	cmpl	$3, %edi
	jl	.LBB0_5
# %bb.1:                                # %entry
	andl	$1, %edi
	jne	.LBB0_2
.LBB0_5:                                # %if.end
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB0_2:                                # %if.then
	.cfi_def_cfa_offset 32
	movl	$8, %edi
	callq	__cxa_allocate_exception@PLT
	movq	%rax, %rbx
.Ltmp0:
	movl	$4, %edi
	callq	_Znwm@PLT
.Ltmp1:
# %bb.3:                                # %invoke.cont
	movl	$0, (%rax)
	movq	%rax, (%rbx)
	movq	_ZTIPi@GOTPCREL(%rip), %rsi
	movq	%rbx, %rdi
	xorl	%edx, %edx
	callq	__cxa_throw@PLT
.LBB0_4:                                # %lpad
.Ltmp2:
	movq	%rax, %r14
	movq	%rbx, %rdi
	callq	__cxa_free_exception@PLT
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end0:
	.size	_Z3bari, .Lfunc_end0-_Z3bari
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Lfunc_end0-.Ltmp1             #   Call between .Ltmp1 and .Lfunc_end0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2
                                        # -- End function
	.text
	.globl	_Z10throw_testiPPc              # -- Begin function _Z10throw_testiPPc
	.p2align	4, 0x90
	.type	_Z10throw_testiPPc,@function
_Z10throw_testiPPc:                     # @_Z10throw_testiPPc
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception1
# %bb.0:                                # %entry
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	cmpl	$99, %edi
	je	.LBB1_7
# %bb.2:                                # %if.end
	movq	%rsi, %r15
	movl	%edi, %r14d
	cmpl	$2, %edi
	movl	$10, %eax
	movl	$5000, %r12d                    # imm = 0x1388
	cmovgeq	%rax, %r12
	xorl	%r13d, %r13d
	xorl	%ebx, %ebx
	.p2align	4, 0x90
.LBB1_3:                                # %for.body
                                        # =>This Inner Loop Header: Depth=1
.Ltmp3:
	movl	%r14d, %edi
	callq	_Z3bari@PLT
.Ltmp4:
.LBB1_4:                                # %for.inc
                                        #   in Loop: Header=BB1_3 Depth=1
	addq	%rbx, %r13
	incq	%rbx
	cmpq	%rbx, %r12
	jne	.LBB1_3
	jmp	.LBB1_6
.LBB1_7:
	xorl	%r13d, %r13d
  jmp .LBB1_8
.LBB1_5:                                # %lpad
                                        #   in Loop: Header=BB1_3 Depth=1
.Ltmp5:
	movq	%rax, %rdi
	callq	__cxa_begin_catch@PLT
	callq	__cxa_end_catch@PLT
	jmp	.LBB1_4
.LBB1_6:                                # %for.cond.cleanup
	movl	%r14d, %eax
	orl	$2, %eax
	cmpl	$7, %eax
	jne	.LBB1_9
  jmp .LBB1_7
.LBB1_8:                                # %cleanup21
	movq	%r13, %rax
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB1_9:                                # %if.end8
	.cfi_def_cfa_offset 48
	leal	-103(%r14), %eax
	cmpl	$101, %eax
	jb	.LBB1_7
# %bb.11:                               # %if.end12
	cmpq	$0, (%r15)
	je	.LBB1_7
# %bb.12:                               # %if.end15
	addl	$-13, %r14d
	xorl	%eax, %eax
	cmpl	$11, %r14d
	cmovbq	%rax, %r13
	jmp	.LBB1_8
.Lfunc_end1:
	.size	_Z10throw_testiPPc, .Lfunc_end1-_Z10throw_testiPPc
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table1:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Ltmp3-.Lfunc_begin1           # >> Call Site 1 <<
	.uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
	.uleb128 .Ltmp5-.Lfunc_begin1           #     jumps to .Ltmp5
	.byte	1                               #   On action: 1
	.uleb128 .Ltmp4-.Lfunc_begin1           # >> Call Site 2 <<
	.uleb128 .Lfunc_end1-.Ltmp4             #   Call between .Ltmp4 and .Lfunc_end1
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end1:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase0:
	.p2align	2
                                        # -- End function
	.text
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	_Z10throw_testiPPc@PLT
	xorl	%ecx, %ecx
	testq	%rax, %rax
	sete	%cl
	movl	%ecx, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"clang version 15.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZTIPi
