## Check that AArch64 mapping symbols ($x) at a function entry are not treated
## as function symbols when the symbol table is rewritten: they must keep a zero
## size and must not get split fragment (.cold.N) symbols, otherwise the output
## cannot be read back by BOLT ("parent function not found for $x.cold.0").

# REQUIRES: system-linux

# RUN: %clang %s %cflags -Wl,-q -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --split-functions \
# RUN:   --split-all-cold --enable-bat
# RUN: llvm-readelf -sW %t.bolt | FileCheck %s --implicit-check-not='$x.cold'

## The functions and their cold fragments are emitted, and $x stays a
## zero-sized mapping symbol.
# CHECK-DAG: [[FOO_ADDR:[0-9a-f]+]] 0 NOTYPE LOCAL DEFAULT [[FOO_SECTION:[0-9]+]] $x{{$}}
# CHECK-DAG: [[BAR_ADDR:[0-9a-f]+]] 0 NOTYPE LOCAL DEFAULT [[BAR_SECTION:[0-9]+]] $x{{$}}
# CHECK-DAG: [[FOO_ADDR]] {{[0-9]+}} FUNC GLOBAL DEFAULT [[FOO_SECTION]] foo{{$}}
# CHECK-DAG: FUNC LOCAL DEFAULT {{[0-9]+}} foo.cold.0{{$}}
# CHECK-DAG: [[BAR_ADDR]] {{[0-9]+}} FUNC GLOBAL DEFAULT [[BAR_SECTION]] bar{{$}}
# CHECK-DAG: FUNC LOCAL DEFAULT {{[0-9]+}} bar.cold.0{{$}}

## The output can be read back.
# RUN: echo 'B 0 0 1 0' > %t.preagg
# RUN: perf2bolt %t.bolt -p %t.preagg --pa -o %t.bolt.fdata 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CHECK-READ
# CHECK-READ-NOT: BOLT-ERROR
# CHECK-READ: PERF2BOLT: wrote

## Each function is in its own section, so each gets a $x mapping symbol at its
## entry, which makes the parent name of $x.cold.0 ambiguous on read-back.
  .section .text.foo,"ax",@progbits
  .globl foo
  .type foo, %function
foo:
.foo_entry:
# FDATA: 1 foo #.foo_entry# 10
  cmp x0, #0
  b.eq .Lfoo_cold
  mov x0, #1
  ret
.Lfoo_cold:
  mov x0, #2
  ret
  .size foo, .-foo

  .section .text.bar,"ax",@progbits
  .globl bar
  .type bar, %function
bar:
.bar_entry:
# FDATA: 1 bar #.bar_entry# 10
  cmp x0, #0
  b.eq .Lbar_cold
  mov x0, #3
  ret
.Lbar_cold:
  mov x0, #4
  ret
  .size bar, .-bar

  .text
  .globl main
  .type main, %function
main:
  mov x0, #0
  ret
  .size main, .-main

## Force relocation mode.
.reloc 0, R_AARCH64_NONE
