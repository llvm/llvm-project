# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe --data %t.fdata -o %t --lite=0 -v=2 \
# RUN:   --cmov-conversion --cmov-conversion-misprediction-threshold=-1 \
# RUN:   --cmov-conversion-bias-threshold=-1 --print-all | FileCheck %s
# CHECK: BOLT-INFO: CMOVConversion: CmovInHotPath, converted static 1/1
# CHECK: BOLT-INFO: CMOVConversion: CmovNotInHotPath, converted static 1/1
# CHECK: BOLT-INFO: CMOVConversion: MaxIndex, converted static 1/1
# CHECK: BOLT-INFO: CMOVConversion: MaxIndex_unpredictable, converted static 1/1
# CHECK: BOLT-INFO: CMOVConversion: MaxValue, converted static 1/1
# CHECK: BOLT-INFO: CMOVConversion: BinarySearch, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: Transform, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: test_cmov_memoperand, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: test_cmov_memoperand_unpredictable, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: test_cmov_memoperand_in_group, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: test_cmov_memoperand_in_group2, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: test_cmov_memoperand_conflicting_dir, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: test_cmov_memoperand_in_group_reuse_for_addr, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: test_cmov_memoperand_in_group_reuse_for_addr2, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: test_cmov_memoperand_in_group_reuse_for_addr3, converted static 0/0
# CHECK: BOLT-INFO: CMOVConversion: test_memoperand_loop, converted static 1/1
# CHECK: BOLT-INFO: CMOVConversion: CmovBackToBack, converted static 2/2
# CHECK: BOLT-INFO: CMOVConversion total: converted static 8/8

	.globl _start
_start:
	.globl	CmovInHotPath                   # -- Begin function CmovInHotPath
	.p2align	4, 0x90
	.type	CmovInHotPath,@function
CmovInHotPath:                          # @CmovInHotPath
# CHECK-LABEL: Binary Function "CmovInHotPath" after CMOV conversion
# FDATA: 0 [unknown] 0 1 CmovInHotPath 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	testl	%edi, %edi
	jle	LBB0_5
# %bb.1:                                # %for.body.preheader
	movl	%edi, %r8d
	xorl	%edi, %edi
# FDATA: 0 [unknown] 0 1 CmovInHotPath #LBB0_2# 1 2
LBB0_2:                                # %for.body
	movl	(%rcx,%rdi,4), %eax
	leal	1(%rax), %r9d
	imull	%esi, %eax
	movl	$10, %r10d
	cmpl	%edx, %eax
# CHECK:      cmpl %edx, %eax
# CHECK-NEXT: cmovlel %r9d, %r10d
LBB0_2_br:
	jg	LBB0_4
# FDATA: 1 CmovInHotPath #LBB0_2_br# 1 CmovInHotPath #LBB0_3# 1 2
# FDATA: 1 CmovInHotPath #LBB0_2_br# 1 CmovInHotPath #LBB0_4# 1 2
# %bb.3:                                # %for.body
LBB0_3:
	movl	%r9d, %r10d
LBB0_4:                                # %for.body
	imull	%r9d, %r10d
	movl	%r10d, (%rcx,%rdi,4)
	addq	$1, %rdi
	cmpq	%rdi, %r8
	jne	LBB0_2
LBB0_5:                                # %for.cond.cleanup
	retq
Lfunc_end0:
	.size	CmovInHotPath, Lfunc_end0-CmovInHotPath
	.cfi_endproc
                                        # -- End function
	.globl	CmovNotInHotPath                # -- Begin function CmovNotInHotPath
	.p2align	4, 0x90
	.type	CmovNotInHotPath,@function
CmovNotInHotPath:                       # @CmovNotInHotPath
# CHECK-LABEL: Binary Function "CmovNotInHotPath" after CMOV conversion
# FDATA: 0 [unknown] 0 1 CmovNotInHotPath 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	testl	%edi, %edi
	jle	LBB1_5
# %bb.1:                                # %for.body.preheader
	movl	%edx, %r9d
	movl	%edi, %r10d
	xorl	%edi, %edi
# FDATA: 0 [unknown] 0 1 CmovNotInHotPath #LBB1_2# 1 2
LBB1_2:                                # %for.body
	movl	(%rcx,%rdi,4), %r11d
	movl	%r11d, %eax
	imull	%esi, %eax
	movl	$10, %edx
	cmpl	%r9d, %eax
# CHECK:      cmpl %r9d, %eax
# CHECK-NEXT: cmovlel %r11d, %edx
LBB1_4_br:
	jg	LBB1_4
# FDATA: 1 CmovNotInHotPath #LBB1_4_br# 1 CmovNotInHotPath #LBB1_3# 1 2
# FDATA: 1 CmovNotInHotPath #LBB1_4_br# 1 CmovNotInHotPath #LBB1_4# 1 2
# %bb.3:                                # %for.body
LBB1_3:
	movl	%r11d, %edx
LBB1_4:                                # %for.body
	movl	%edx, (%rcx,%rdi,4)
	movl	(%r8,%rdi,4), %eax
	cltd
	idivl	%r9d
	movl	%eax, (%r8,%rdi,4)
	addq	$1, %rdi
	cmpq	%rdi, %r10
	jne	LBB1_2
LBB1_5:                                # %for.cond.cleanup
	retq
Lfunc_end1:
	.size	CmovNotInHotPath, Lfunc_end1-CmovNotInHotPath
	.cfi_endproc
                                        # -- End function
	.globl	MaxIndex                        # -- Begin function MaxIndex
	.p2align	4, 0x90
	.type	MaxIndex,@function
MaxIndex:                               # @MaxIndex
# CHECK-LABEL: Binary Function "MaxIndex" after CMOV conversion
# FDATA: 0 [unknown] 0 1 MaxIndex 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	xorl	%eax, %eax
	cmpl	$2, %edi
	jl	LBB2_5
# %bb.1:                                # %for.body.preheader
	movl	%edi, %r8d
	xorl	%edi, %edi
	movl	$1, %edx
# FDATA: 0 [unknown] 0 1 MaxIndex #LBB2_2# 1 2
LBB2_2:                                # %for.body
	movl	(%rsi,%rdx,4), %r9d
	movslq	%edi, %rcx
	movl	%edx, %eax
	cmpl	(%rsi,%rcx,4), %r9d
# CHECK:      cmpl	(%rsi,%rcx,4), %r9d
# CHECK-NEXT: cmovlel %edi, %eax
LBB2_2_br:
	jg	LBB2_4
# FDATA: 1 MaxIndex #LBB2_2_br# 1 MaxIndex #LBB2_3# 1 2
# FDATA: 1 MaxIndex #LBB2_2_br# 1 MaxIndex #LBB2_4# 1 2
# %bb.3:                                # %for.body
LBB2_3:
	movl	%edi, %eax
LBB2_4:                                # %for.body
	addq	$1, %rdx
	movl	%eax, %edi
	cmpq	%rdx, %r8
	jne	LBB2_2
LBB2_5:                                # %for.cond.cleanup
	retq
Lfunc_end2:
	.size	MaxIndex, Lfunc_end2-MaxIndex
	.cfi_endproc
                                        # -- End function
	.globl	MaxIndex_unpredictable          # -- Begin function MaxIndex_unpredictable
	.p2align	4, 0x90
	.type	MaxIndex_unpredictable,@function
MaxIndex_unpredictable:                 # @MaxIndex_unpredictable
# CHECK-LABEL: Binary Function "MaxIndex_unpredictable" after CMOV conversion
# FDATA: 0 [unknown] 0 1 MaxIndex_unpredictable 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	xorl	%eax, %eax
	cmpl	$2, %edi
	jl	LBB3_5
# %bb.1:                                # %for.body.preheader
	movl	%edi, %r8d
	xorl	%edi, %edi
	movl	$1, %edx
# FDATA: 0 [unknown] 0 1 MaxIndex_unpredictable #LBB3_2# 1 2
LBB3_2:                                # %for.body
	movl	(%rsi,%rdx,4), %r9d
	movslq	%edi, %rcx
	movl	%edx, %eax
	cmpl	(%rsi,%rcx,4), %r9d
# CHECK:      cmpl	(%rsi,%rcx,4), %r9d
# CHECK-NEXT: cmovlel %edi, %eax
LBB3_2_br:
	jg	LBB3_4
# FDATA: 1 MaxIndex_unpredictable #LBB3_2_br# 1 MaxIndex_unpredictable #LBB3_3# 1 2
# FDATA: 1 MaxIndex_unpredictable #LBB3_2_br# 1 MaxIndex_unpredictable #LBB3_4# 1 2
# %bb.3:                                # %for.body
LBB3_3:
	movl	%edi, %eax
LBB3_4:                                # %for.body
	addq	$1, %rdx
	movl	%eax, %edi
	cmpq	%rdx, %r8
	jne	LBB3_2
LBB3_5:                                # %for.cond.cleanup
	retq
Lfunc_end3:
	.size	MaxIndex_unpredictable, Lfunc_end3-MaxIndex_unpredictable
	.cfi_endproc
                                        # -- End function
	.globl	MaxValue                        # -- Begin function MaxValue
	.p2align	4, 0x90
	.type	MaxValue,@function
MaxValue:                               # @MaxValue
# CHECK-LABEL: Binary Function "MaxValue" after CMOV conversion
# FDATA: 0 [unknown] 0 1 MaxValue 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movl	(%rsi), %ecx
	cmpl	$2, %edi
	jge	LBB4_3
# %bb.1:
LBB4_1:
	movl	%ecx, %eax
LBB4_2:                                # %for.cond.cleanup
	retq
LBB4_3:                                # %for.body.preheader
	movl	%edi, %edi
	movl	$1, %edx
LBB4_4:                                # %for.body
	movl	(%rsi,%rdx,4), %eax
	cmpl	%ecx, %eax
# CHECK:      cmpl	%ecx, %eax
# CHECK-NEXT: cmovlel %ecx, %eax
LBB4_4_br:
	jg	LBB4_6
# FDATA: 1 MaxValue #LBB4_4_br# 1 MaxValue #LBB4_5# 1 2
# FDATA: 1 MaxValue #LBB4_4_br# 1 MaxValue #LBB4_6# 1 2
# %bb.5:                                # %for.body
LBB4_5:
	movl	%ecx, %eax
LBB4_6:                                # %for.body
	addq	$1, %rdx
	movl	%eax, %ecx
	cmpq	%rdx, %rdi
	je	LBB4_2
	jmp	LBB4_4
Lfunc_end4:
	.size	MaxValue, Lfunc_end4-MaxValue
	.cfi_endproc
                                        # -- End function
	.globl	BinarySearch                    # -- Begin function BinarySearch
	.p2align	4, 0x90
	.type	BinarySearch,@function
BinarySearch:                           # @BinarySearch
# CHECK-LABEL: Binary Function "BinarySearch" after CMOV conversion
# FDATA: 0 [unknown] 0 1 BinarySearch 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movl	(%rsi), %eax
	jmp	LBB5_2
LBB5_1:                                # %while.body
	movl	%ecx, %eax
	xorl	%ecx, %ecx
	btl	%eax, %edi
	setae	%cl
	movq	8(%rdx,%rcx,8), %rdx
LBB5_2:                                # %while.body
	movl	(%rdx), %ecx
	cmpl	%ecx, %eax
	ja	LBB5_1
# %bb.3:                                # %while.end
	retq
Lfunc_end5:
	.size	BinarySearch, Lfunc_end5-BinarySearch
	.cfi_endproc
                                        # -- End function
	.globl	Transform                       # -- Begin function Transform
	.p2align	4, 0x90
	.type	Transform,@function
Transform:                              # @Transform
# CHECK-LABEL: Binary Function "Transform" after CMOV conversion
# FDATA: 0 [unknown] 0 1 Transform 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movb	$1, %al
	testb	%al, %al
	jne	LBB6_5
# %bb.1:                                # %while.body.preheader
	movl	%edx, %r8d
	xorl	%esi, %esi
LBB6_2:                                # %while.body
	movslq	%esi, %rsi
	movl	(%rdi,%rsi,4), %eax
	xorl	%edx, %edx
	divl	%r8d
	movl	%eax, %edx
	movl	$11, %eax
	movl	%r8d, %ecx
	cmpl	%r8d, %edx
	ja	LBB6_4
# %bb.3:                                # %while.body
	movl	$22, %eax
	movl	$22, %ecx
LBB6_4:                                # %while.body
	xorl	%edx, %edx
	divl	%ecx
	movl	%edx, (%rdi,%rsi,4)
	addl	$1, %esi
	cmpl	%r9d, %esi
	ja	LBB6_2
LBB6_5:                                # %while.end
	retq
Lfunc_end6:
	.size	Transform, Lfunc_end6-Transform
	.cfi_endproc
                                        # -- End function
	.globl	test_cmov_memoperand            # -- Begin function test_cmov_memoperand
	.p2align	4, 0x90
	.type	test_cmov_memoperand,@function
test_cmov_memoperand:                   # @test_cmov_memoperand
# CHECK-LABEL: Binary Function "test_cmov_memoperand" after CMOV conversion
# FDATA: 0 [unknown] 0 1 test_cmov_memoperand 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%edx, %eax
	cmpl	%esi, %edi
	ja	LBB7_2
# %bb.1:                                # %entry
	movl	(%rcx), %eax
LBB7_2:                                # %entry
	retq
Lfunc_end7:
	.size	test_cmov_memoperand, Lfunc_end7-test_cmov_memoperand
	.cfi_endproc
                                        # -- End function
	.globl	test_cmov_memoperand_unpredictable # -- Begin function test_cmov_memoperand_unpredictable
	.p2align	4, 0x90
	.type	test_cmov_memoperand_unpredictable,@function
test_cmov_memoperand_unpredictable:     # @test_cmov_memoperand_unpredictable
# CHECK-LABEL: Binary Function "test_cmov_memoperand_unpredictable" after CMOV conversion
# FDATA: 0 [unknown] 0 1 test_cmov_memoperand_unpredictable 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%edx, %eax
	cmpl	%esi, %edi
	ja	LBB8_2
# %bb.1:                                # %entry
	movl	(%rcx), %eax
LBB8_2:                                # %entry
	retq
Lfunc_end8:
	.size	test_cmov_memoperand_unpredictable, Lfunc_end8-test_cmov_memoperand_unpredictable
	.cfi_endproc
                                        # -- End function
	.globl	test_cmov_memoperand_in_group   # -- Begin function test_cmov_memoperand_in_group
	.p2align	4, 0x90
	.type	test_cmov_memoperand_in_group,@function
test_cmov_memoperand_in_group:          # @test_cmov_memoperand_in_group
# CHECK-LABEL: Binary Function "test_cmov_memoperand_in_group" after CMOV conversion
# FDATA: 0 [unknown] 0 1 test_cmov_memoperand_in_group 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%edx, %eax
	movl	%edx, %r8d
	cmpl	%esi, %edi
	ja	LBB9_2
# %bb.1:                                # %entry
	movl	(%rcx), %r8d
	movl	%edi, %eax
	movl	%esi, %edx
LBB9_2:                                # %entry
	addl	%r8d, %eax
	addl	%edx, %eax
	retq
Lfunc_end9:
	.size	test_cmov_memoperand_in_group, Lfunc_end9-test_cmov_memoperand_in_group
	.cfi_endproc
                                        # -- End function
	.globl	test_cmov_memoperand_in_group2  # -- Begin function test_cmov_memoperand_in_group2
	.p2align	4, 0x90
	.type	test_cmov_memoperand_in_group2,@function
test_cmov_memoperand_in_group2:         # @test_cmov_memoperand_in_group2
# CHECK-LABEL: Binary Function "test_cmov_memoperand_in_group2" after CMOV conversion
# FDATA: 0 [unknown] 0 1 test_cmov_memoperand_in_group2 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%edx, %eax
	movl	%edx, %r8d
	cmpl	%esi, %edi
	jbe	LBB10_2
# %bb.1:                                # %entry
	movl	(%rcx), %r8d
	movl	%edi, %eax
	movl	%esi, %edx
LBB10_2:                               # %entry
	addl	%r8d, %eax
	addl	%edx, %eax
	retq
Lfunc_end10:
	.size	test_cmov_memoperand_in_group2, Lfunc_end10-test_cmov_memoperand_in_group2
	.cfi_endproc
                                        # -- End function
	.globl	test_cmov_memoperand_conflicting_dir # -- Begin function test_cmov_memoperand_conflicting_dir
	.p2align	4, 0x90
	.type	test_cmov_memoperand_conflicting_dir,@function
test_cmov_memoperand_conflicting_dir:   # @test_cmov_memoperand_conflicting_dir
# CHECK-LABEL: Binary Function "test_cmov_memoperand_conflicting_dir" after CMOV conversion
# FDATA: 0 [unknown] 0 1 test_cmov_memoperand_conflicting_dir 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	cmpl	%esi, %edi
	movl	(%rcx), %eax
	cmoval	%edx, %eax
	cmoval	(%r8), %edx
	addl	%edx, %eax
	retq
Lfunc_end11:
	.size	test_cmov_memoperand_conflicting_dir, Lfunc_end11-test_cmov_memoperand_conflicting_dir
	.cfi_endproc
                                        # -- End function
	.globl	test_cmov_memoperand_in_group_reuse_for_addr # -- Begin function test_cmov_memoperand_in_group_reuse_for_addr
	.p2align	4, 0x90
	.type	test_cmov_memoperand_in_group_reuse_for_addr,@function
test_cmov_memoperand_in_group_reuse_for_addr: # @test_cmov_memoperand_in_group_reuse_for_addr
# CHECK-LABEL: Binary Function "test_cmov_memoperand_in_group_reuse_for_addr" after CMOV conversion
# FDATA: 0 [unknown] 0 1 test_cmov_memoperand_in_group_reuse_for_addr 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%edi, %eax
	cmpl	%esi, %edi
	ja	LBB12_2
# %bb.1:                                # %entry
	movl	(%rcx), %eax
LBB12_2:                               # %entry
	retq
Lfunc_end12:
	.size	test_cmov_memoperand_in_group_reuse_for_addr, Lfunc_end12-test_cmov_memoperand_in_group_reuse_for_addr
	.cfi_endproc
                                        # -- End function
	.globl	test_cmov_memoperand_in_group_reuse_for_addr2 # -- Begin function test_cmov_memoperand_in_group_reuse_for_addr2
	.p2align	4, 0x90
	.type	test_cmov_memoperand_in_group_reuse_for_addr2,@function
test_cmov_memoperand_in_group_reuse_for_addr2: # @test_cmov_memoperand_in_group_reuse_for_addr2
# CHECK-LABEL: Binary Function "test_cmov_memoperand_in_group_reuse_for_addr2" after CMOV conversion
# FDATA: 0 [unknown] 0 1 test_cmov_memoperand_in_group_reuse_for_addr2 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%edi, %eax
	cmpl	%esi, %edi
	ja	LBB13_2
# %bb.1:                                # %entry
	movq	(%rcx), %rax
	movl	(%rax), %eax
LBB13_2:                               # %entry
	retq
Lfunc_end13:
	.size	test_cmov_memoperand_in_group_reuse_for_addr2, Lfunc_end13-test_cmov_memoperand_in_group_reuse_for_addr2
	.cfi_endproc
                                        # -- End function
	.globl	test_cmov_memoperand_in_group_reuse_for_addr3 # -- Begin function test_cmov_memoperand_in_group_reuse_for_addr3
	.p2align	4, 0x90
	.type	test_cmov_memoperand_in_group_reuse_for_addr3,@function
test_cmov_memoperand_in_group_reuse_for_addr3: # @test_cmov_memoperand_in_group_reuse_for_addr3
# CHECK-LABEL: Binary Function "test_cmov_memoperand_in_group_reuse_for_addr3" after CMOV conversion
# FDATA: 0 [unknown] 0 1 test_cmov_memoperand_in_group_reuse_for_addr3 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movl	%edi, %eax
	cmpl	%esi, %edi
	ja	LBB14_2
# %bb.1:                                # %entry
	movl	(%rcx), %eax
LBB14_2:                               # %entry
	retq
Lfunc_end14:
	.size	test_cmov_memoperand_in_group_reuse_for_addr3, Lfunc_end14-test_cmov_memoperand_in_group_reuse_for_addr3
	.cfi_endproc
                                        # -- End function
	.globl	test_memoperand_loop            # -- Begin function test_memoperand_loop
	.p2align	4, 0x90
	.type	test_memoperand_loop,@function
test_memoperand_loop:                   # @test_memoperand_loop
# CHECK-LABEL: Binary Function "test_memoperand_loop" after CMOV conversion
# FDATA: 0 [unknown] 0 1 test_memoperand_loop 0 1 2
	.cfi_startproc
# %bb.0:                                # %entry
	movq	begin@GOTPCREL(%rip), %r8
	movq	(%r8), %rax
	movq	end@GOTPCREL(%rip), %rcx
	movq	(%rcx), %rdx
	xorl	%esi, %esi
	movq	%rax, %rcx
LBB15_1:                               # %loop.body
	addq	$8, %rcx
	cmpq	%rdx, %rcx
	ja	LBB15_3
# %bb.2:                                # %loop.body
	movq	(%r8), %rcx
LBB15_3:                               # %loop.body
	movl	%edi, (%rcx)
	addq	$8, %rcx
	cmpq	%rdx, %rcx
# CHECK:      movl	%edi, (%rcx)
# CHECK-NEXT: addq	$0x8, %rcx
# CHECK-NEXT: cmpq	%rdx, %rcx
# CHECK-NEXT: cmovbeq %rax, %rcx
LBB15_3_br:
	ja	LBB15_5
# FDATA: 1 test_memoperand_loop #LBB15_3_br# 1 test_memoperand_loop #LBB15_4# 1 2
# FDATA: 1 test_memoperand_loop #LBB15_3_br# 1 test_memoperand_loop #LBB15_5# 1 2
# %bb.4:                                # %loop.body
LBB15_4:
	movq	%rax, %rcx
LBB15_5:                               # %loop.body
	movl	%edi, (%rcx)
	addl	$1, %esi
	cmpl	$1024, %esi                     # imm = 0x400
	jl	LBB15_1
# %bb.6:                                # %exit
	retq
Lfunc_end15:
	.size	test_memoperand_loop, Lfunc_end15-test_memoperand_loop
	.cfi_endproc
                                        # -- End function
	.globl	CmovBackToBack                   # -- Begin function CmovBackToBack
	.p2align	4, 0x90
	.type	CmovBackToBack,@function
CmovBackToBack:                          # @CmovBackToBack
# CHECK-LABEL: Binary Function "CmovBackToBack" after CMOV conversion
# FDATA: 0 [unknown] 0 1 CmovBackToBack 0 1 2
	.cfi_startproc
	testl	%edi, %edi
	jle	LBB16_5
	movl	%edi, %r8d
	xorl	%edi, %edi
# FDATA: 0 [unknown] 0 1 CmovBackToBack #LBB16_2# 1 2
LBB16_2:                                # %for.body
	movl	(%rcx,%rdi,4), %eax
	leal	1(%rax), %r9d
	imull	%esi, %eax
	movl	$10, %r10d
	cmpl	%edx, %eax
# CHECK:      cmpl %edx, %eax
# CHECK-NEXT: cmovlel %r9d, %r10d
LBB16_2_br:
	jg	LBB16_4
# FDATA: 1 CmovBackToBack #LBB16_2_br# 1 CmovBackToBack #LBB16_3# 1 2
# FDATA: 1 CmovBackToBack #LBB16_2_br# 1 CmovBackToBack #LBB16_4# 1 2
LBB16_3:
	movl	%r9d, %r10d
LBB16_4:                                # %for.body
# CHECK-NEXT: cmovlel %r9d, %r10d
LBB16_6_br:
	jg	LBB16_8
# FDATA: 1 CmovBackToBack #LBB16_6_br# 1 CmovBackToBack #LBB16_7# 1 2
# FDATA: 1 CmovBackToBack #LBB16_6_br# 1 CmovBackToBack #LBB16_8# 1 2
LBB16_7:
	movl	%r9d, %r10d
LBB16_8:                                # %for.body
	imull	%r9d, %r10d
	movl	%r10d, (%rcx,%rdi,4)
	addq	$1, %rdi
	cmpq	%rdi, %r8
	jne	LBB16_2
LBB16_5:                                # %for.cond.cleanup
	retq
Lfunc_end16:
	.size	CmovBackToBack, Lfunc_end16-CmovBackToBack
	.cfi_endproc
                                        # -- End function
  .data
  .globl begin
begin:
  .quad 0xdeadbeef
  .globl end
end:
  .quad 0xfaceb00c

