## Ensures that a call continuation fallthrough count is set when using
## pre-aggregated perf data.

# RUN: %clang %cflags -fpic -shared -xc /dev/null -o %t.so
## Link against a DSO to ensure PLT entries.
# RUN: %clangxx %cxxflags %s %t.so -o %t -Wl,-q -nostdlib
# Trace to a call continuation, not a landing pad/entry point
# RUN: link_fdata %s %t %t.pa-base PREAGG-BASE
# Trace from a return to a landing pad/entry point call continuation
# RUN: link_fdata %s %t %t.pa-ret PREAGG-RET
# Trace from an external location to a landing pad/entry point call continuation
# RUN: link_fdata %s %t %t.pa-ext PREAGG-EXT
# Return trace to a landing pad/entry point call continuation
# RUN: link_fdata %s %t %t.pa-pret PREAGG-PRET
# External return to a landing pad/entry point call continuation
# RUN: link_fdata %s %t %t.pa-eret PREAGG-ERET
# RUN-DISABLED: link_fdata %s %t %t.pa-plt PREAGG-PLT

# RUN: llvm-strip --strip-unneeded %t -o %t.strip
# RUN: llvm-objcopy --remove-section=.eh_frame %t.strip %t.noeh

## Check pre-aggregated traces attach call continuation fallthrough count
## in the basic case (not an entry point, not a landing pad).
# RUN: llvm-bolt %t.noeh --pa -p %t.pa-base -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK-BASE

## Check pre-aggregated traces from a return attach call continuation
## fallthrough count to secondary entry point (unstripped)
# RUN: llvm-bolt %t --pa -p %t.pa-ret -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK-ATTACH
## Check pre-aggregated traces from a return attach call continuation
## fallthrough count to landing pad (stripped, landing pad)
# RUN: llvm-bolt %t.strip --pa -p %t.pa-ret -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK-ATTACH

## Check pre-aggregated traces from external location don't attach call
## continuation fallthrough count to secondary entry point (unstripped)
# RUN: llvm-bolt %t --pa -p %t.pa-ext -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK-SKIP
## Check pre-aggregated traces from external location don't attach call
## continuation fallthrough count to landing pad (stripped, landing pad)
# RUN: llvm-bolt %t.strip --pa -p %t.pa-ext -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK-SKIP

## Check pre-aggregated return traces from external location attach call
## continuation fallthrough count to secondary entry point (unstripped)
# RUN: llvm-bolt %t --pa -p %t.pa-pret -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK-ATTACH
## Check pre-aggregated return traces from external location attach call
## continuation fallthrough count to landing pad (stripped, landing pad)
# RUN: llvm-bolt %t.strip --pa -p %t.pa-pret -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK-ATTACH

## Same for external return type
# RUN: llvm-bolt %t --pa -p %t.pa-eret -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK-ATTACH
# RUN: llvm-bolt %t.strip --pa -p %t.pa-eret -o %t.out \
# RUN:   --print-cfg --print-only=main | FileCheck %s --check-prefix=CHECK-ATTACH

## Check pre-aggregated traces don't report zero-sized PLT fall-through as
## invalid trace
# RUN-DISABLED: llvm-bolt %t.strip --pa -p %t.pa-plt -o %t.out | FileCheck %s \
# RUN-DISABLED:   --check-prefix=CHECK-PLT
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
# PREAGG-PLT: T #Ltmp0_br# #puts@plt# #puts@plt# 3
## Target is an external-origin call continuation
# PREAGG-BASE: T X:0 #Ltmp1# #Ltmp4_br# 2
# CHECK-BASE:      callq puts@PLT
# CHECK-BASE-NEXT: count: 2

Ltmp1:
	movq	-0x10(%rbp), %rax
	movq	0x8(%rax), %rdi
	movl	%eax, -0x14(%rbp)

Ltmp4:
	cmpl	$0x0, -0x14(%rbp)
Ltmp4_br:
	je	Ltmp0

	movl	$0xa, -0x18(%rbp)
	callq	foo
## Target is a binary-local call continuation
# PREAGG-RET: T #Lfoo_ret# #Ltmp3# #Ltmp3_br# 1
## Target is a secondary entry point (unstripped) or a landing pad (stripped)
# PREAGG-EXT: T X:0 #Ltmp3# #Ltmp3_br# 1
## Pre-aggregated return trace
# PREAGG-PRET: R X:0 #Ltmp3# #Ltmp3_br# 1
## External return
# PREAGG-ERET: r #Ltmp3# #Ltmp3_br# 1

# CHECK-ATTACH:      callq foo
# CHECK-ATTACH-NEXT: count: 1
# CHECK-SKIP:        callq foo
# CHECK-SKIP-NEXT:   count: 0

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
