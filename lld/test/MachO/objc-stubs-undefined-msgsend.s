# REQUIRES: aarch64

## _objc_msgSend is needed by the objc stubs but not provided by any input.
## This must produce an undefined symbol error instead of crashing.

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: not %lld -arch arm64 -o %t.out %t.o 2>&1 | FileCheck %s
# RUN: not %lld -arch arm64 -o %t.out %t.o -objc_stubs_small 2>&1 | FileCheck %s
# RUN: not %lld -arch arm64 -o %t.out %t.o -no_fixup_chains 2>&1 | FileCheck %s

## Class stubs need it too. With both kinds of stubs it is still set up once, so
## the reference is listed once.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin --defsym CLASS=1 %s -o %t-class.o
# RUN: not %lld -arch arm64 -o %t.out %t-class.o 2>&1 | FileCheck %s
# RUN: not %lld -arch arm64 -o %t.out %t-class.o -objc_stubs_small 2>&1 | FileCheck %s

# CHECK:      error: undefined symbol: _objc_msgSend
# CHECK-NEXT: >>> referenced by lazy binding (normally in libobjc.dylib)
# CHECK-NOT:  >>> referenced by

.section __TEXT,__text
.globl _main
_main:
  bl _objc_msgSend$foo
.ifdef CLASS
  bl _objc_msgSendClass$foo$_OBJC_CLASS_$_Foo
.endif
  ret

.ifdef CLASS
.data
.globl _OBJC_CLASS_$_Foo
_OBJC_CLASS_$_Foo:
  .quad 0
.endif

.subsections_via_symbols
