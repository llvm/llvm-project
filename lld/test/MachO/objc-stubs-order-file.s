# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/a.s -o %t/a.o
# RUN: echo _hot > %t/order

# RUN: %lld -arch arm64 -e _main -U _objc_msgSend -o %t/base.out %t/a.o \
# RUN:   -objc_stubs_small
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --macho \
# RUN:   %t/base.out | FileCheck %s --check-prefix=BASE

# RUN: %lld -arch arm64 -e _main -U _objc_msgSend -o %t/ordered.out %t/a.o \
# RUN:   -objc_stubs_small -order_file %t/order
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --macho \
# RUN:   %t/ordered.out | FileCheck %s --check-prefix=ORDERED

# BASE:      Contents of (__TEXT,__objc_stubs) section
# BASE-NEXT: _objc_msgSend$cold:
# BASE:      _objc_msgSend$hot:

# ORDERED:      Contents of (__TEXT,__objc_stubs) section
# ORDERED-NEXT: _objc_msgSend$hot:
# ORDERED:      _objc_msgSend$cold:

#--- a.s
.text
.globl _cold
.p2align 2
_cold:
  bl _objc_msgSend$cold
  ret

.globl _hot
.p2align 2
_hot:
  bl _objc_msgSend$hot
  ret

.globl _main
.p2align 2
_main:
  bl _cold
  bl _hot
  ret

.subsections_via_symbols
