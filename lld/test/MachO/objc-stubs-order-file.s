# REQUIRES: aarch64

## __objc_stubs is synthetic, so the input section sorting does not reach its
## entries. Check that they are instead ordered by the priority of the sections
## that call them.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o

## Without an order file the stubs keep the order they were interned in.
# RUN: %lld -arch arm64 -e _main -U _objc_msgSend -o %t/base.out %t/a.o \
# RUN:   -objc_stubs_small
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --macho \
# RUN:   %t/base.out | FileCheck %s --check-prefix=BASE

## _hot is ordered first, so the stub it calls moves ahead of the others, which
## keep their relative order.
# RUN: %lld -arch arm64 -e _main -U _objc_msgSend -o %t/ordered.out %t/a.o \
# RUN:   -objc_stubs_small -order_file %t/order
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --macho \
# RUN:   %t/ordered.out | FileCheck %s --check-prefix=ORDERED

# RUN: %lld -arch arm64 -e _main -U _objc_msgSend -o %t/fast.out %t/a.o \
# RUN:   -objc_stubs_fast -order_file %t/order
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --macho \
# RUN:   %t/fast.out | FileCheck %s --check-prefix=ORDERED

## Callers are recorded after ICF. _dup is identical to _hot and folds with it,
## so the priority has to reach the stub through the surviving section.
# RUN: %lld -arch arm64 -e _main -U _objc_msgSend -o %t/icf.out %t/a.o \
# RUN:   -objc_stubs_small -order_file %t/order --icf=all
# RUN: llvm-nm --numeric-sort %t/icf.out | FileCheck %s --check-prefix=FOLDED
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --macho \
# RUN:   %t/icf.out | FileCheck %s --check-prefix=ORDERED

## Check that _dup folded into _hot.
# FOLDED:      [[ADDR:[0-9a-f]+]] T _dup
# FOLDED-NEXT: [[ADDR]] T _hot

# BASE:      Contents of (__TEXT,__objc_stubs) section
# BASE-NEXT: _objc_msgSend$cold:
# BASE:      _objc_msgSend$hot:
# BASE:      _objc_msgSend$mild:

# ORDERED:      Contents of (__TEXT,__objc_stubs) section
# ORDERED-NEXT: _objc_msgSend$hot:
# ORDERED:      _objc_msgSend$cold:
# ORDERED:      _objc_msgSend$mild:

#--- a.s
.text
.globl _cold
.p2align 2
_cold:
  bl _objc_msgSend$cold
  ret

.globl _mild
.p2align 2
_mild:
  bl _objc_msgSend$mild
  ret

.globl _hot
.p2align 2
_hot:
  bl _objc_msgSend$hot
  ret

.globl _dup
.p2align 2
_dup:
  bl _objc_msgSend$hot
  ret

.globl _main
.p2align 2
_main:
  bl _cold
  bl _mild
  bl _hot
  ret

.subsections_via_symbols

#--- order
_hot
