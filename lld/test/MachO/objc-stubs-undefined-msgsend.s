# REQUIRES: aarch64

## _objc_msgSend is needed by the objc stubs but not provided by any input.
## This must produce an undefined symbol error instead of crashing.

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: not %lld -arch arm64 -o %t.out %t.o 2>&1 | FileCheck %s
# RUN: not %lld -arch arm64 -o %t.out %t.o -objc_stubs_small 2>&1 | FileCheck %s
# RUN: not %lld -arch arm64 -o %t.out %t.o -no_fixup_chains 2>&1 | FileCheck %s

# CHECK:      error: undefined symbol: _objc_msgSend
# CHECK-NEXT: >>> referenced by lazy binding (normally in libobjc.dylib)

.section __TEXT,__text
.globl _main
_main:
  bl _objc_msgSend$foo
  ret

.subsections_via_symbols
