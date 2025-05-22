# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld --pie %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe -o %t.out --data %t.fdata --split-functions --split-eh \
# RUN:   --split-all-cold --print-after-lowering  --print-only=_start 2>&1 \
# RUN:   | FileCheck %s

## _start has two landing pads: one hot and one cold. Hence, BOLT will introduce
## a landing pad trampoline. However, the trampoline code will make the main
## split fragment larger than the whole function before split. Then BOLT will
## undo the splitting and remove the trampoline.

# CHECK: Binary Function "_start"
# CHECK: IsSplit :
# CHECK-SAME: 0

## Check that a landing pad trampoline was created, but contains no instructions
## and falls though to the real landing pad.

# CHECK: {{^[^[:space:]]+}} (0 instructions
# CHECK-NEXT: Landing Pad{{$}}
# CHECK: Exec Count
# CHECK-SAME: : 0
# CHECK: Successors:
# CHECK-SAME: [[LP:[^[:space:]]+]]
# CHECK-EMPTY:
# CHECK-NEXT: [[LP]]

  .text
	.global foo
  .type foo, %function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo

  .globl _start
  .type _start, %function
_start:
# FDATA: 0 [unknown] 0 1 _start 0 1 100
.Lfunc_begin0:
  .cfi_startproc
	.cfi_lsda 27, .Lexception0
	call foo
.Ltmp0:
	call foo
.Ltmp1:
  ret

## Cold landing pad.
.LLP1:
  ret

## Hot landing pad.
LLP0:
# FDATA: 0 [unknown] 0 1 _start #LLP0# 1 100
	ret

  .cfi_endproc
.Lfunc_end0:
  .size _start, .-_start

## EH table.
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0  # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0         #   Call between .Lfunc_begin0 and .Ltmp0
	.uleb128 LLP0-.Lfunc_begin0					#   jumps to LLP0
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0         # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .LLP1-.Lfunc_begin0          #     jumps to .LLP1
	.byte	0                               #   On action: cleanup
.Lcst_end0:

