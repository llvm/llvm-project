# REQUIRES: aarch64

## The three-hop cases exercise targets above and below their callers.
##
## Final layout:
##
##   0                 ~+96 MiB       ~+192 MiB      ~+288 MiB      ~+384 MiB
##   |                     |               |              |              |
## _three_hop_down/   extenders/      extenders/    extenders/    _three_hop/
## _main              _boundary0      _one_hop      _two_hop      _downward_main
##
## Expected routing (`.iN` := `.island.N`):
##
##   -arm64-islands-max-hops 0:
##     _main --> {one,two,three}.thunk.0 ==> each target
##     down target <== down.thunk.0 <-- _downward_main
##
##   -arm64-islands-max-hops 2:
##     _main --> one.i0 --> _one_hop
##     _main --> two.i0 --> two.i1 --> _two_hop
##     _main --> three.thunk.0 ==> _three_hop
##     _boundary0 --> three.i1 --> three.i2 --> _three_hop
##     _one_hop ----------------> three.i2 --> _three_hop
##     down target <== down.thunk.0 <-- _downward_main
##
##   -arm64-islands-max-hops 4294967295
##   -arm64-thunks-max-hops  0:
##     _main --> three.i0 --> three.i1 --> three.i2 --> _three_hop
##     down target <-- down.i0 <-- down.i1 <-- down.i2 <-- down main

# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/input.o

# RUN: %lld -arch arm64 -dylib -o %t/thunks %t/input.o \
# RUN:   -arm64-islands-max-hops 0
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/thunks \
# RUN:   | FileCheck %s --check-prefix=THUNKS
# RUN: rm %t/thunks

# RUN: %lld -arch arm64 -dylib -o %t/island-hops-2 %t/input.o \
# RUN:   -arm64-islands-max-hops 2
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/island-hops-2 \
# RUN:   | FileCheck %s --check-prefix=ISLAND-HOPS-2
# RUN: rm %t/island-hops-2

# RUN: %lld -arch arm64 -dylib -o %t/islands-hops-inf %t/input.o \
# RUN:   -arm64-islands-max-hops 4294967295 -arm64-thunks-max-hops 0
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/islands-hops-inf \
# RUN:   | FileCheck %s --check-prefix=ISLANDS-HOPS-INF
# RUN: rm %t/islands-hops-inf

## No configured extender can serve these branches when both are disabled.
# RUN: not %lld -arch arm64 -dylib -o %t/disabled %t/input.o \
# RUN:   -arm64-islands-max-hops 0 -arm64-thunks-max-hops 0 2>&1 \
# RUN:   | FileCheck %s --check-prefix=DISABLED
# DISABLED: error: cannot place branch extension for _one_hop

## A single island cannot reach the two-hop destination without a thunk.
# RUN: not %lld -arch arm64 -dylib -o %t/one-hop %t/input.o \
# RUN:   -arm64-islands-max-hops 1 -arm64-thunks-max-hops 0 2>&1 \
# RUN:   | FileCheck %s --check-prefix=ONE-HOP
# ONE-HOP: error: cannot place branch extension for _two_hop

## Invalid option values must fail.
# RUN: not %lld -arch arm64 -dylib -o %t/invalid %t/input.o \
# RUN:   -arm64-islands-max-hops -1 2>&1 | FileCheck %s --check-prefix=INVALID-ISLANDS
# INVALID-ISLANDS: error: -arm64-islands-max-hops: expected an integer in the range [0, 4294967295], but got '-1'

# RUN: not %lld -arch arm64 -dylib -o %t/invalid %t/input.o \
# RUN:   -arm64-thunks-max-hops inf 2>&1 | FileCheck %s --check-prefix=INVALID-THUNKS
# INVALID-THUNKS: error: -arm64-thunks-max-hops: expected an integer in the range [0, 4294967295], but got 'inf'

# THUNKS-LABEL: <_main>:
# THUNKS-NEXT: bl 0x[[#%x, ONE:]] <_one_hop.thunk.0>
# THUNKS-NEXT: bl 0x[[#%x, TWO:]] <_two_hop.thunk.0>
# THUNKS-NEXT: bl 0x[[#%x, THREE:]] <_three_hop.thunk.0>
# THUNKS: [[#ONE]] <_one_hop.thunk.0>:
# THUNKS-NEXT: adrp x16
# THUNKS-NEXT: add x16, x16
# THUNKS-NEXT: br x16
# THUNKS: [[#TWO]] <_two_hop.thunk.0>:
# THUNKS-NEXT: adrp x16
# THUNKS-NEXT: add x16, x16
# THUNKS-NEXT: br x16
# THUNKS: [[#THREE]] <_three_hop.thunk.0>:
# THUNKS-NEXT: adrp x16
# THUNKS-NEXT: add x16, x16
# THUNKS-NEXT: br x16
# THUNKS: [[#%x, DOWN:]] <_three_hop_down.thunk.0>:
# THUNKS-NEXT: adrp x16
# THUNKS-NEXT: add x16, x16
# THUNKS-NEXT: br x16
# THUNKS-LABEL: <_downward_main>:
# THUNKS-NEXT: bl 0x[[#DOWN]] <_three_hop_down.thunk.0>

# ISLAND-HOPS-2-LABEL: <_main>:
# ISLAND-HOPS-2-NEXT: bl 0x[[#%x, ONE:]] <_one_hop.island.0>
# ISLAND-HOPS-2-NEXT: bl 0x[[#%x, TWO0:]] <_two_hop.island.0>
# ISLAND-HOPS-2-NEXT: bl 0x[[#%x, THREE:]] <_three_hop.thunk.0>
# ISLAND-HOPS-2: [[#ONE]] <_one_hop.island.0>:
# ISLAND-HOPS-2-NEXT: b 0x{{[0-9a-f]+}} <_one_hop>
# ISLAND-HOPS-2: [[#TWO0]] <_two_hop.island.0>:
# ISLAND-HOPS-2-NEXT: b 0x[[#%x, TWO1:]] <_two_hop.island.1>
# ISLAND-HOPS-2: [[#THREE]] <_three_hop.thunk.0>:
# ISLAND-HOPS-2-NEXT: adrp x16
# ISLAND-HOPS-2-NEXT: add x16, x16
# ISLAND-HOPS-2-NEXT: br x16
# ISLAND-HOPS-2-LABEL: <_boundary0>:
# ISLAND-HOPS-2-NEXT: bl 0x[[#%x, THREE1:]] <_three_hop.island.1>
# ISLAND-HOPS-2: [[#TWO1]] <_two_hop.island.1>:
# ISLAND-HOPS-2-NEXT: b 0x{{[0-9a-f]+}} <_two_hop>
# ISLAND-HOPS-2: [[#THREE1]] <_three_hop.island.1>:
# ISLAND-HOPS-2-NEXT: b 0x[[#%x, THREE2:]] <_three_hop.island.2>
# ISLAND-HOPS-2-LABEL: <_one_hop>:
# ISLAND-HOPS-2-NEXT: bl 0x[[#THREE2]] <_three_hop.island.2>
# ISLAND-HOPS-2: [[#THREE2]] <_three_hop.island.2>:
# ISLAND-HOPS-2-NEXT: b 0x{{[0-9a-f]+}} <_three_hop>
# ISLAND-HOPS-2: [[#%x, DOWN:]] <_three_hop_down.thunk.0>:
# ISLAND-HOPS-2-NEXT: adrp x16
# ISLAND-HOPS-2-NEXT: add x16, x16
# ISLAND-HOPS-2-NEXT: br x16
# ISLAND-HOPS-2-LABEL: <_downward_main>:
# ISLAND-HOPS-2-NEXT: bl 0x[[#DOWN]] <_three_hop_down.thunk.0>

# ISLANDS-HOPS-INF-LABEL: <_main>:
# ISLANDS-HOPS-INF-NEXT: bl 0x[[#%x, ONE:]] <_one_hop.island.0>
# ISLANDS-HOPS-INF-NEXT: bl 0x[[#%x, TWO0:]] <_two_hop.island.0>
# ISLANDS-HOPS-INF-NEXT: bl 0x[[#%x, THREE0:]] <_three_hop.island.0>
# ISLANDS-HOPS-INF: [[#ONE]] <_one_hop.island.0>:
# ISLANDS-HOPS-INF-NEXT: b 0x{{[0-9a-f]+}} <_one_hop>
# ISLANDS-HOPS-INF: [[#TWO0]] <_two_hop.island.0>:
# ISLANDS-HOPS-INF-NEXT: b 0x[[#%x, TWO1:]] <_two_hop.island.1>
# ISLANDS-HOPS-INF: [[#THREE0]] <_three_hop.island.0>:
# ISLANDS-HOPS-INF-NEXT: b 0x[[#%x, THREE1:]] <_three_hop.island.1>
# ISLANDS-HOPS-INF: [[#%x, DOWN0:]] <_three_hop_down.island.0>:
# ISLANDS-HOPS-INF-NEXT: b 0x{{[0-9a-f]+}} <_three_hop_down>
# ISLANDS-HOPS-INF-LABEL: <_boundary0>:
# ISLANDS-HOPS-INF-NEXT: bl 0x[[#THREE1]] <_three_hop.island.1>
# ISLANDS-HOPS-INF: [[#TWO1]] <_two_hop.island.1>:
# ISLANDS-HOPS-INF-NEXT: b 0x{{[0-9a-f]+}} <_two_hop>
# ISLANDS-HOPS-INF: [[#THREE1]] <_three_hop.island.1>:
# ISLANDS-HOPS-INF-NEXT: b 0x[[#%x, THREE2:]] <_three_hop.island.2>
# ISLANDS-HOPS-INF: [[#%x, DOWN1:]] <_three_hop_down.island.1>:
# ISLANDS-HOPS-INF-NEXT: b 0x[[#DOWN0]] <_three_hop_down.island.0>
# ISLANDS-HOPS-INF-LABEL: <_one_hop>:
# ISLANDS-HOPS-INF-NEXT: bl 0x[[#THREE2]] <_three_hop.island.2>
# ISLANDS-HOPS-INF: [[#THREE2]] <_three_hop.island.2>:
# ISLANDS-HOPS-INF-NEXT: b 0x{{[0-9a-f]+}} <_three_hop>
# ISLANDS-HOPS-INF: [[#%x, DOWN2:]] <_three_hop_down.island.2>:
# ISLANDS-HOPS-INF-NEXT: b 0x[[#DOWN1]] <_three_hop_down.island.1>
# ISLANDS-HOPS-INF-LABEL: <_downward_main>:
# ISLANDS-HOPS-INF-NEXT: bl 0x[[#DOWN2]] <_three_hop_down.island.2>
# ISLANDS-HOPS-INF-NOT: .thunk.

.subsections_via_symbols
.text

.globl _three_hop_down
.p2align 2
_three_hop_down:
  ret

.globl _main
.p2align 2
_main:
  bl _one_hop
  bl _two_hop
  bl _three_hop
  ret

.space 0x6000000
.globl _boundary0
_boundary0:
  bl _three_hop
  ret

.space 0x6000000
.globl _one_hop
_one_hop:
  bl _three_hop
  ret

.space 0x6000000
.globl _two_hop
_two_hop:
  ret

.space 0x6000000
.globl _three_hop
_three_hop:
  ret

.globl _downward_main
.p2align 2
_downward_main:
  bl _three_hop_down
  ret
