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
# RUN: llvm-nm -m %t.exe | FileCheck %s --check-prefix=EXE
# RUN: %lld -arch arm64 -lSystem --icf=all %t.o -o %t.icf.exe -map %t/icf.map
# RUN: FileCheck %s --input-file %t/icf.map --check-prefix=ICF
# RUN: %lld -arch arm64 -lSystem --icf=safe_thunks %t.o -o %t.safe.exe -map %t/safe.map
# RUN: FileCheck %s --input-file %t/safe.map --check-prefix=SAFE-THUNKS

#--- test.s
.subsections_via_symbols

.addrsig
.addrsig_sym _cold
.addrsig_sym _cold_addrsig
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

.globl _cold_addrsig
.p2align 2
.desc _cold_addrsig, 0x400
_cold_addrsig:
  add x0, x1, x2
  add x3, x4, x5
  ret

.globl _cold_unordered
.p2align 2
.desc _cold_unordered, 0x400
_cold_unordered:
  add x0, x1, x3
  ret

.globl _main
.p2align 2
_main:
  bl _normal
  bl _cold
  bl _cold_addrsig
  bl _ordered
  bl _cold_unordered
  ret

## Basic N_COLD_FUNC support.
# NOORDER: <_normal>:
# NOORDER: <_ordered>:
# NOORDER: <_main>:
# NOORDER: <_cold>:
# NOORDER: <_cold_unordered>:

## Ordered symbols should come before unordered cold symbols.
# ORDER-HOT: <_ordered>:
# ORDER-HOT: <_normal>:
# ORDER-HOT: <_main>:
# ORDER-HOT: <_cold>:
# ORDER-HOT: <_cold_unordered>:

## Cold attribute should not change the ordering of order-file symbols.
# ORDER-COLD: <_cold>:
# ORDER-COLD: <_ordered>:
# ORDER-COLD: <_normal>:
# ORDER-COLD: <_main>:
# ORDER-COLD: <_cold_unordered>:

## Check that N_COLD_FUNC is NOT preserved in the output executable.
# EXE: (__TEXT,__text) external _cold

## ICF + N_COLD_FUNC: _cold, _normal, and _ordered have identical bodies.
## _cold is the master; since _normal (non-cold) is folded into _cold,
## _cold's isCold is unset and it stays in the hot region.
## _cold_unordered has a different body and stays cold.
# ICF-LABEL: # Symbols:
# ICF:       _cold
# ICF:       _normal
# ICF:       _ordered
# ICF:       _main
# ICF:       _cold_unordered

## With safe_thunks, _cold, _cold_addrsig, and _normal are keepUnique. _cold
## appears first in input order so it becomes the master. Since _normal
## (non-cold) is folded into _cold, _cold's isCold is unset. _normal gets a
## non-cold thunk. _cold_addrsig gets a cold thunk. _cold_unordered stays cold
## and anchors the cold region.
# SAFE-THUNKS-LABEL: # Symbols:
# SAFE-THUNKS:       0x0000000C {{.*}} _cold
# SAFE-THUNKS:       0x00000000 {{.*}} _ordered
# SAFE-THUNKS:       {{.*}} _main
# SAFE-THUNKS:       0x00000004 {{.*}} _normal
# SAFE-THUNKS:       {{.*}} _cold_unordered
# SAFE-THUNKS:       0x00000004 {{.*}} _cold_addrsig

#--- ord-hot
_ordered

#--- ord-cold
_cold
_ordered
