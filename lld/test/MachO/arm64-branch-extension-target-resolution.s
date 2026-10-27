# REQUIRES: aarch64

## Test how target-address availability affects branch extension planning.
## _far_nocode is in the current segment, so its address is known even though
## its section has no instruction attributes. _main needs a thunk to reach it;
## _tail_caller can call it directly.
## _far_hib is in a later segment, so its address is unknown while planning
## __TEXT. Both callers use pessimistic single-hop thunks, including the nearby
## _tail_caller. A larger configured hop limit must still permit this fallback.

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -o %t %t.o -arm64-islands-max-hops 0
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s
# RUN: %lld -arch arm64 -o %t %t.o -arm64-islands-max-hops 1
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s
## A larger hop limit still permits a single thunk to an unknown cross-segment
## target. Pessimistically choose the widest enabled extender for that hop.
# RUN: %lld -arch arm64 -o %t %t.o -arm64-thunks-max-hops 2
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s

# CHECK-LABEL: Disassembly of section __TEXT,__text:
# CHECK-LABEL: <_main>:
# CHECK-NEXT: bl 0x[[#%x, NOCODE_THUNK:]] <_far_nocode.thunk.0>
# CHECK-NEXT: bl 0x[[#%x, HIB_THUNK:]] <_far_hib.thunk.0>
# CHECK-NEXT: ret
# CHECK-DAG: [[#NOCODE_THUNK]] <_far_nocode.thunk.0>:
# CHECK-DAG: [[#HIB_THUNK]] <_far_hib.thunk.0>:
# CHECK-LABEL: <_tail_caller>:
# CHECK-NEXT: bl 0x{{[0-9a-f]+}} <_far_nocode{{(\.thunk\.[0-9]+)?}}>
# CHECK-NEXT: bl 0x{{[0-9a-f]+}} <_far_hib{{(\.thunk\.[0-9]+)?}}>
# CHECK-NEXT: ret

.text
.globl _main
.p2align 2
_main:
  bl _far_nocode
  bl _far_hib
  ret

_spacer:
.space 0x8100000

.globl _tail_caller
.p2align 2
_tail_caller:
  bl _far_nocode
  bl _far_hib
  ret

.section __TEXT,__far
.globl _far_nocode
.p2align 2
_far_nocode:
  ret

.section __HIB,__text,regular,pure_instructions
.globl _far_hib
.p2align 2
_far_hib:
  ret

.subsections_via_symbols
