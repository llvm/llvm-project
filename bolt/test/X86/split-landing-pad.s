# This test reproduces the case where C++ exception handling is used and split
# function optimization is enabled. In particular, function foo is splitted
# to two fragments:
#    foo: contains 2 try blocks, which invokes bar to throw exception
#    foo.cold.1: contains 2 corresponding catch blocks (landing pad)
#
# Similar to split jump table, split landing pad target to different fragment.
# This test is written to ensure BOLT safely handle these targets, e.g., by
# marking them as non-simple.
#
# Steps to write this test:
# - Create a copy of Inputs/src/unreachable.cpp
# - Simplify bar(), focus on throw an exception
# - Create the second switch case in foo() to have multiple landing pads
# - Compile with clang++ to .s
# - Move landing pad code from foo to foo.cold.1
# - Ensure that all landing pads can be reached normally
#
# Additional details:
# .gcc_except_table specify the landing pads for try blocks
#    LPStart = 255 (omit), which means LPStart = foo start
# Landing pads .Ltmp2 and .Ltmp5 in call site record are offset to foo start.


# REQUIRES: system-linux
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clang++ %cxxflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt -v=3 %t.exe -o %t.out 2>&1 | FileCheck %s

# CHECK: BOLT-WARNING: Ignoring foo
# CHECK: BOLT-WARNING: Ignoring foo.cold.1
# CHECK: BOLT-WARNING: skipped 2 functions due to cold fragments

	.text
	.globl	bar                             # -- Begin function bar
	.p2align	4, 0x90
	.type	bar,@function
bar:                                    # @bar
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	cmpq	$34, -8(%rbp)
	jne	.LBB0_2
# %bb.1:                                # %if.then
	movl	$4, %edi
	callq	__cxa_allocate_exception@PLT
	movq	%rax, %rdi
	movl	$0, (%rdi)
	movq	_ZTIi@GOTPCREL(%rip), %rsi
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	__cxa_throw@PLT
.LBB0_2:                                # %if.else
	movl	$8, %edi
	callq	__cxa_allocate_exception@PLT
	movq	%rax, %rdi
	movq	$0, (%rdi)
	movq	_ZTIDn@GOTPCREL(%rip), %rsi
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	__cxa_throw@PLT
.Lfunc_end0:
	.size	bar, .Lfunc_end0-bar
	.cfi_endproc
                                        # -- End function
	.globl	foo                             # -- Begin function foo
	.p2align	4, 0x90
	.type	foo,@function
foo:                                    # @foo
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$48, %rsp
	movq	%rdi, -16(%rbp)
	movq	-16(%rbp), %rdi
.Ltmp0:
	callq	bar
.Ltmp1:
	jmp	.LBB1_1
.LBB1_1:                                # %invoke.cont
	jmp	.LBB1_5
.LBB1_5:                                # %try.cont
	movq	-16(%rbp), %rdi
	addq	$34, %rdi
.Ltmp3:
	callq	bar
.Ltmp4:
	jmp	.LBB1_6
.LBB1_6:                                # %invoke.cont2
	jmp	.LBB1_10
.LBB1_10:                               # %try.cont8
	movq	$0, -8(%rbp)
.LBB1_11:                               # %return
	movq	-8(%rbp), %rax
	addq	$48, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.LBB1_12:                               # %eh.resume
	.cfi_def_cfa %rbp, 16
	movq	-24(%rbp), %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end1:
	.size	foo, .Lfunc_end1-foo
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table1:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 1 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	1                               #   On action: 1
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp3-.Ltmp1                  #   Call between .Ltmp1 and .Ltmp3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
	.uleb128 .Ltmp5-.Lfunc_begin0           #     jumps to .Ltmp5
	.byte	1                               #   On action: 1
	.uleb128 .Ltmp4-.Lfunc_begin0           # >> Call Site 4 <<
	.uleb128 .Lfunc_end1-.Ltmp4             #   Call between .Ltmp4 and .Lfunc_end1
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2
                                        # >> Catch TypeInfos <<
.Ltmp6:                                 # TypeInfo 1
	.long	.L_ZTIi.DW.stub-.Ltmp6
.Lttbase0:
	.p2align	2
                                        # -- End function


        .text
	.globl	foo.cold.1              # -- Begin function foo.cold.1
	.p2align	4, 0x90
	.type	foo.cold.1,@function
foo.cold.1:                                    # @foo.cold.1
.Lfunc_begin3:
	.cfi_startproc
.Ltmp2:
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -24(%rbp)
	movl	%eax, -28(%rbp)
# %bb.3:                                # %catch.dispatch
	movl	-28(%rbp), %eax
	movl	$1, %ecx
	cmpl	%ecx, %eax
	jne	.LBB1_12
# %bb.4:                                # %catch
	movq	-24(%rbp), %rdi
	callq	__cxa_begin_catch@PLT
	movl	(%rax), %eax
	movl	%eax, -32(%rbp)
	movq	$0, -8(%rbp)
	callq	__cxa_end_catch@PLT
	jmp	.LBB1_11
.Ltmp5:
	movq	%rax, %rcx
	movl	%edx, %eax
	movq	%rcx, -24(%rbp)
	movl	%eax, -28(%rbp)
# %bb.8:                                # %catch.dispatch3
	movl	-28(%rbp), %eax
	movl	$1, %ecx
	cmpl	%ecx, %eax
	jne	.LBB1_12
# %bb.9:                                # %catch6
	movq	-24(%rbp), %rdi
	callq	__cxa_begin_catch@PLT
	movl	(%rax), %eax
	movl	%eax, -36(%rbp)
	movq	$0, -8(%rbp)
	callq	__cxa_end_catch@PLT
	jmp	.LBB1_11
.Lfunc_end3:
	.size	foo.cold.1, .Lfunc_end3-foo.cold.1
	.cfi_endproc


	.text
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	xorl	%eax, %eax
	movl	%eax, %edi
	callq	foo
                                        # kill: def $eax killed $eax killed $rax
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.data
	.p2align	3
.L_ZTIi.DW.stub:
	.quad	_ZTIi
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym bar
	.addrsig_sym __cxa_allocate_exception
	.addrsig_sym __cxa_throw
	.addrsig_sym foo
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym __cxa_begin_catch
	.addrsig_sym __cxa_end_catch
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZTIi
	.addrsig_sym _ZTIDn
        .addrsig_sym foo.cold.1
        .addrsig_sym __cxa_begin_catch
        .addrsig_sym __cxa_end_catch
