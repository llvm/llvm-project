## Ensures that a call continuation fallthrough count is set when using
## pre-aggregated perf data.

# RUN: %clang %cflags -fpic -shared -xc /dev/null -o %t.so
## Link against a DSO to ensure PLT entries.
# RUN: %clangxx %cxxflags %s %t.so -o %t -Wl,-q -nostdlib
# RUN: link_fdata %s %t %t.pat PREAGGT1
# RUN: link_fdata %s %t %t.pat2 PREAGGT2
# RUN: link_fdata %s %t %t.patplt PREAGGPLT

# RUN: llvm-strip --strip-unneeded %t -o %t.strip
# RUN: llvm-objcopy --remove-section=.eh_frame %t.strip %t.noeh

## Check pre-aggregated traces attach call continuation fallthrough count
# RUN: llvm-bolt %t.noeh --pa -p %t.pat -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s

## Check pre-aggregated traces don't attach call continuation fallthrough count
## to secondary entry point (unstripped)
# RUN: llvm-bolt %t --pa -p %t.pat2 -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK3
## Check pre-aggregated traces don't attach call continuation fallthrough count
## to landing pad (stripped, LP)
# RUN: llvm-bolt %t.strip --pa -p %t.pat2 -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK3

## Check pre-aggregated traces don't report zero-sized PLT fall-through as
## invalid trace
# RUN: llvm-bolt %t.strip --pa -p %t.patplt -o %t.out | FileCheck %s \
# RUN:   --check-prefix=CHECK-PLT
# CHECK-PLT: traces mismatching disassembled function contents: 0

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
Ltmp0_br:
	callq	puts@PLT
## Check PLT traces are accepted
# PREAGGPLT: T #Ltmp0_br# #puts@plt# #puts@plt# 3
## Target is an external-origin call continuation
# PREAGGT1: T X:0 #Ltmp1# #Ltmp4_br# 2
# CHECK:      callq puts@PLT
# CHECK-NEXT: count: 2

Ltmp1:
	movq	-0x10(%rbp), %rax
	movq	0x8(%rax), %rdi
	movl	%eax, -0x14(%rbp)

Ltmp4:
	cmpl	$0x0, -0x14(%rbp)
Ltmp4_br:
	je	Ltmp0
# CHECK2:      je .Ltmp0
# CHECK2-NEXT: count: 3

	movl	$0xa, -0x18(%rbp)
	callq	foo
## Target is a binary-local call continuation
# PREAGGT1: T #Lfoo_ret# #Ltmp3# #Ltmp3_br# 1
# CHECK:      callq foo
# CHECK-NEXT: count: 1

## PLT call continuation fallthrough spanning the call
# CHECK2:      callq foo
# CHECK2-NEXT: count: 3

## Target is a secondary entry point (unstripped) or a landing pad (stripped)
# PREAGGT2: T X:0 #Ltmp3# #Ltmp3_br# 2
# CHECK3:      callq foo
# CHECK3-NEXT: count: 0

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
