# REQUIRES: aarch64

## A direct call that is in range in the reservation layout can become
## out of range in the exact layout. The planner must mark it as requiring
## an extender so that subsequent passes do not choose a direct call again.
## Planning the extender can then grow the reservation or yield a valid layout.
##
## Here, removing unused reserved space moves the caller to a lower address.
## The callee stays at the same address because its alignment padding grows
## to absorb the removed space. This increases the branch displacement beyond
## the architectural limit, even though it fits in the reservation layout.

## Input order in __text (A/B/C are boundaries):
##
##   main.o                       callers.o                 near.o
##   _trigger -> _far_target     call -> _thunk_target      _near_caller
##   | padding (~64 MiB) | A | padding / call / padding | B | padding | C |
##                            <------ ~128 MiB ------>      ~128 MiB
##
## _far_target is absolute at 0x140002e4; its initial route uses islands at
## A and B. Later route use only one thunk at A.
## _thunk_target is absolute at 4 GiB and always requires a thunk at B.
## Pass   Reserved bytes A/B   Exact proposal at A/B        Result
##   1          0 / 0         island / island + thunk      grow to 4 / 16
##   2          4 / 16        thunk / thunk                grow to 12 / 16
##   3         12 / 16        thunk / thunk                promote near call
##   4         12 / 16        thunk / thunk, island at C    valid
##
## Final offsets relative to _near_caller (which follows main.o and callers.o):
##   caller + 0 --> island at +0x7ffff84 --> target at +0x8000000

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/callers.s -o %t/callers.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/near.s -o %t/near.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/target.s -o %t/target.o
# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -o %t/out \
# RUN:   %t/main.o %t/callers.o %t/near.o %t/target.o \
# RUN:   -arm64-islands-max-hops 2
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/out \
# RUN:   | FileCheck %s --check-prefix=DIS
# RUN: llvm-nm -n %t/out | FileCheck %s --check-prefix=LAYOUT

# DIS-LABEL: <_near_caller>:
# DIS: bl 0x[[#%x, ISLAND:]] <_near_target.island.0>
# DIS: [[#ISLAND]] <_near_target.island.0>:
# DIS-NEXT: b 0x{{[0-9a-f]+}} <_near_target>

# LAYOUT: [[#%x, CALLER:]] T _near_caller
# LAYOUT: [[#CALLER + 0x7ffff84]] t _near_target.island.0
# LAYOUT: [[#CALLER + 0x8000000]] T _near_target

#--- main.s
  .subsections_via_symbols
  .text
.globl _trigger
.p2align 2
_trigger:
  bl _far_target

.globl _padding
_padding:
  .space 0x4000068

#--- callers.s
  .text

## This always emit a thunk.
  .space 0x3fffffc
  bl _thunk_target
  .space 0x3fffffc

#--- near.s
  .text
.globl _near_caller
_near_caller:
  bl _near_target
  .space 0x7ffff80

#--- target.s
  .text

.p2align 7
.globl _near_target
_near_target:
  ret

.globl _far_target
_far_target = 0x140002e4

.globl _thunk_target
_thunk_target = 0x100000000
