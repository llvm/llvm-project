# REQUIRES: aarch64

## Hand-written assembly without instruction attributes still needs branch
## extension. Both targets and the caller are in an attribute-free section:
## extend the far call to _target_lo and keep the nearby _target_hi call direct.

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -o %t %t.o -arm64-islands-max-hops 0
# RUN: llvm-objdump --syms --no-print-imm-hex -D --no-show-raw-insn %t \
# RUN:   | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE:
# CHECK-DAG: [[#%x, THUNK:]] l O __TEXT,__asm .hidden _target_lo.thunk.0
# CHECK-DAG: [[#%x, TARGET_HI:]] g O __TEXT,__asm _target_hi
# CHECK-LABEL: Disassembly of section __TEXT,__asm:
# CHECK-DAG: [[#THUNK]] <_target_lo.thunk.0>:
# CHECK-DAG: [[#TARGET_HI]] <_target_hi>:
# CHECK-LABEL: <_main>:
# CHECK-NEXT: bl 0x[[#THUNK]] <_target_lo.thunk.0>
# CHECK-NEXT: bl 0x[[#TARGET_HI]] <_target_hi>
# CHECK-NEXT: ret

.section __TEXT,__asm
.globl _target_lo
.p2align 2
_target_lo:
  ret

_spacer:
.space 0x8100000

.globl _target_hi
.p2align 2
_target_hi:
  ret

.globl _main
.p2align 2
_main:
  bl _target_lo
  bl _target_hi
  ret

.subsections_via_symbols
