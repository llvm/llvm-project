# REQUIRES: aarch64

## Test that when two weak definitions of the same symbol exist and either one
## is marked N_COLD_FUNC, the merged symbol is treated as cold. This matches the
## behavior of both ld-prime and ld64.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/weak-cold.s -o %t/weak-cold.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/weak-noncold.s -o %t/weak-noncold.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/weak-main.s -o %t/weak-main.o

## Link order: non-cold first, cold second. The non-cold def wins but cold |= true.
# RUN: %lld -arch arm64 -lSystem %t/weak-noncold.o %t/weak-cold.o %t/weak-main.o -o %t/weak-nc-c.exe
# RUN: llvm-objdump -d %t/weak-nc-c.exe | FileCheck %s

## Link order: cold first, non-cold second. The cold def wins and cold |= false.
# RUN: %lld -arch arm64 -lSystem %t/weak-cold.o %t/weak-noncold.o %t/weak-main.o -o %t/weak-c-nc.exe
# RUN: llvm-objdump -d %t/weak-c-nc.exe | FileCheck %s

## In both link orders, _weakfn ends up cold and is placed after _main.
# CHECK: <_main>:
# CHECK: <_weakfn>:

#--- weak-cold.s
.subsections_via_symbols
.text
.globl _weakfn
.weak_definition _weakfn
.p2align 2
.desc _weakfn, 0x400
_weakfn:
  add x0, x1, x2
  ret

#--- weak-noncold.s
.subsections_via_symbols
.text
.globl _weakfn
.weak_definition _weakfn
.p2align 2
_weakfn:
  add x0, x1, x2
  ret

#--- weak-main.s
.subsections_via_symbols
.text
.globl _main
.p2align 2
_main:
  bl _weakfn
  ret
