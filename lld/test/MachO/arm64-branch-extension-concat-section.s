# REQUIRES: aarch64

## Branch extension planning supports sections in any segment, not just __TEXT.
## Check that a thunk in __HIB is laid out and written.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -o %t %t.o -segprot __HIB rx rx -arm64-islands-max-hops 0
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s

# CHECK-LABEL: Disassembly of section __HIB,__text:
# CHECK-LABEL: <_caller>:
# CHECK-NEXT: bl 0x[[#%x, THUNK:]] <__HIB._target.thunk.0>
# CHECK-NEXT: ret
# CHECK: [[#THUNK]] <__HIB._target.thunk.0>:
# CHECK-NEXT: adrp x16,
# CHECK-NEXT: add x16, x16,
# CHECK-NEXT: br x16
# CHECK-LABEL: <_target>:
# CHECK-NEXT: ret

.text
.globl _main
.p2align 2
_main:
  ret

.section __HIB,__text,regular,pure_instructions
.globl _caller
.p2align 2
_caller:
  bl _target
  ret

_spacer:
.space 0x8100000

.globl _target
.p2align 2
_target:
  ret

.subsections_via_symbols
