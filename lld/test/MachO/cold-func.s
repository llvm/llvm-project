# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/test.s -o %t.o
# RUN: %lld -arch arm64 -lSystem %t.o -o %t.noorder.exe
# RUN: llvm-objdump -d %t.noorder.exe | FileCheck %s --check-prefix=NOORDER
# RUN: %lld -arch arm64 -lSystem %t.o -o %t.order-hot.exe -order_file %t/ord-hot
# RUN: llvm-objdump -d %t.order-hot.exe | FileCheck %s --check-prefix=ORDER-HOT
# RUN: %lld -arch arm64 -lSystem %t.o -o %t.order-cold.exe -order_file %t/ord-cold
# RUN: llvm-objdump -d %t.order-cold.exe | FileCheck %s --check-prefix=ORDER-COLD
# RUN: %lld -arch arm64 -lSystem %t.o -o %t.exe
# RUN: llvm-objdump --syms %t.exe | FileCheck %s --check-prefix=EXE
# RUN: %lld -arch arm64 -lSystem --icf=all %t.o -o %t.icf.exe -map %t/icf.map
# RUN: FileCheck %s --input-file %t/icf.map --check-prefix=ICF
# RUN: %lld -arch arm64 -lSystem --icf=safe_thunks %t.o -o %t.safe.exe -map %t/safe.map
# RUN: FileCheck %s --input-file %t/safe.map --check-prefix=SAFE-THUNKS

#--- test.s
.subsections_via_symbols

## Mark _cold and _normal as address-significant for safe_thunks testing.
.addrsig
.addrsig_sym _cold
.addrsig_sym _normal

.text

.globl _cold
.p2align 2
.desc _cold, 0x400
_cold:
  add x0, x1, x2
  add x3, x4, x5
  ret

.globl _normal
.p2align 2
_normal:
  add x0, x1, x2
  add x3, x4, x5
  ret

.globl _ordered
.p2align 2
_ordered:
  add x0, x1, x2
  add x3, x4, x5
  ret

.globl _main
.p2align 2
_main:
  bl _normal
  bl _cold
  bl _ordered
  ret

## Basic N_COLD_FUNC support.
# NOORDER: <_normal>:
# NOORDER: <_ordered>:
# NOORDER: <_main>:
# NOORDER: <_cold>:

## Ordered symbols should come before unordered cold symbols.
# ORDER-HOT: <_ordered>:
# ORDER-HOT: <_normal>:
# ORDER-HOT: <_main>:
# ORDER-HOT: <_cold>:

## Cold attribute should not change the ordering of order-file symbols.
# ORDER-COLD: <_cold>:
# ORDER-COLD: <_ordered>:
# ORDER-COLD: <_normal>:
# ORDER-COLD: <_main>:

## Check that N_COLD_FUNC is NOT preserved in the output executable.
# EXE:      SYMBOL TABLE:
# EXE-NOT:  0400 {{.*}} _cold
# EXE:      {{.*}} g     F __TEXT,__text _cold

## ICF + N_COLD_FUNC: _cold, _normal, and _ordered have identical bodies.
## After folding, the non-cold copy should be the master so the folded body
## is not the cold region (after _main).
# ICF-LABEL: # Symbols:
# ICF-DAG:   _normal
# ICF-DAG:   _cold
# ICF-DAG:   _ordered
# ICF:       _main

## With safe_thunks, _cold and _normal are keepUnique. The non-cold _normal
## should be chosen as the master, appearing before _main. _cold gets a thunk
## placed in the cold region.
# SAFE-THUNKS-LABEL: # Symbols:
# SAFE-THUNKS:       0x0000000C {{.*}} _normal
# SAFE-THUNKS:       0x00000000 {{.*}} _ordered
# SAFE-THUNKS:       0x00000010 {{.*}} _main
# SAFE-THUNKS:       0x00000004 {{.*}} _cold

#--- ord-hot
_ordered

#--- ord-cold
_cold
_ordered
