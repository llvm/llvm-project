# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o

# RUN: %lld -arch arm64 -lSystem -U _objc_msgSend -o %t.out %t.o
# RUN: llvm-nm %t.out | FileCheck %s
# RUN: %lld -arch arm64 -lSystem -U _objc_msgSend -dead_strip -o %t.out %t.o
# RUN: llvm-nm %t.out | FileCheck %s --check-prefix=DEAD

# CHECK: _foo
# CHECK: _objc_msgSend$length

# DEAD-NOT: _foo
# DEAD-NOT: _objc_msgSend$length

.section __TEXT,__text

.globl _foo
_foo:
  bl  _objc_msgSend$length
  ret

.globl _main
_main:
  ret

.subsections_via_symbols
