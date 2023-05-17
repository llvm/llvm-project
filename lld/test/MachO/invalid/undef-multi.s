# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: not %lld -arch arm64 %t.o -o /dev/null 2>&1 | FileCheck -DFILE=%t.o %s

# CHECK: error: undefined symbol: _undef
# CHECK-NEXT: >>> referenced by [[FILE]]:(symbol _main+0x0)
# CHECK-NEXT: >>> referenced by [[FILE]]:(symbol _foo+0x0)
# CHECK-NEXT: >>> referenced by [[FILE]]:(symbol _bar+0x0)
# CHECK-NEXT: >>> referenced 1 more times

.globl _main
_main:
    b _undef

.globl _foo
_foo:
    b _undef

.global _bar
_bar:
    b _undef

.globl _baz
_baz:
    b _undef

.subsections_via_symbols
