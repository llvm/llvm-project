## Ensures that a call continuation fallthrough count is set when using
## pre-aggregated perf data.

# RUN: %clangxx %cxxflags %s -o %t -Wl,-q -nostdlib
# RUN: link_fdata %s %t %t.pa1 PREAGG
# RUN: link_fdata %s %t %t.pa2 PREAGG2
# RUN: link_fdata %s %t %t.pa3 PREAGG3
# RUN: link_fdata %s %t %t.pa4 PREAGG4

## Check normal case: fallthrough is not LP or secondary entry.
# RUN: llvm-strip --strip-unneeded %t -o %t.exe
# RUN: llvm-bolt %t.exe --pa -p %t.pa1 -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s

## Check that getFallthroughsInTrace correctly handles a trace starting at plt
## call continuation
# RUN: llvm-bolt %t.exe --pa -p %t.pa2 -o %t.out2 \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK2

## Check that we don't treat secondary entry points as call continuation sites.
# RUN: llvm-bolt %t --pa -p %t.pa3 -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK3

## Check fallthrough to a landing pad case.
# RUN: llvm-bolt %t.exe --pa -p %t.pa4 -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK4

  .globl foo
  .type foo, %function
foo:
	pushq	%rbp
	movq	%rsp, %rbp
	popq	%rbp
Lfoo_ret:
	retq
.size foo, .-foo

  .globl main
  .type main, %function
main:
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
	pushq	%rbp
	movq	%rsp, %rbp
	subq	$0x20, %rsp
	movl	$0x0, -0x4(%rbp)
	movl	%edi, -0x8(%rbp)
	movq	%rsi, -0x10(%rbp)
	callq	puts@PLT
## Target is a call continuation
# PREAGG: B X:0 #Ltmp1# 2 0
# CHECK:      callq puts@PLT
# CHECK-NEXT: count: 2

Ltmp1:
	movq	-0x10(%rbp), %rax
	movq	0x8(%rax), %rdi
	movl	%eax, -0x14(%rbp)

Ltmp4:
	cmpl	$0x0, -0x14(%rbp)
	je	Ltmp0
# CHECK2:      je .Ltmp0
# CHECK2-NEXT: count: 3

	movl	$0xa, -0x18(%rbp)
	callq	foo
## Target is a call continuation
# PREAGG: B #Lfoo_ret# #Ltmp3# 1 0
# CHECK:      callq foo
# CHECK-NEXT: count: 1

## PLT call continuation fallthrough spanning the call
# PREAGG2: F #Ltmp1# #Ltmp3_br# 3
# CHECK2:      callq foo
# CHECK2-NEXT: count: 3

## Target is a secondary entry point
# PREAGG3: B X:0 #Ltmp3# 2 0
# CHECK3:      callq foo
# CHECK3-NEXT: count: 0

## Target is a landing pad
# PREAGG4: B X:0 #Ltmp3# 2 0
# CHECK4:      callq puts@PLT
# CHECK4-NEXT: count: 0

Ltmp3:
	cmpl	$0x0, -0x18(%rbp)
Ltmp3_br:
	jmp	Ltmp2

Ltmp2:
	movl	-0x18(%rbp), %eax
	addl	$-0x1, %eax
	movl	%eax, -0x18(%rbp)
	jmp	Ltmp3
	jmp	Ltmp4
	jmp	Ltmp1

Ltmp0:
	xorl	%eax, %eax
	addq	$0x20, %rsp
	popq	%rbp
	retq
.Lfunc_end0:
  .cfi_endproc
.size main, .-main

	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Lfunc_end0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Lfunc_end0
	.uleb128 Ltmp3-.Lfunc_begin0           #     jumps to Ltmp3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3, 0x0
	.type	DW.ref.__gxx_personality_v0,@object
