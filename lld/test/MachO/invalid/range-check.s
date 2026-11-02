# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: %lld -dylib %t/bar.o -o %t/libbar.dylib
# RUN: not %lld -lSystem -o /dev/null %t/libbar.dylib %t/test.o 2>&1 | FileCheck %s

# CHECK-DAG: error: {{.*}}test.o:(symbol _main+0xd): relocation UNSIGNED is out of range: [[#]] is not in [0, 4294967295]; references _foo
# CHECK-DAG: error: {{.*}}test.o:(symbol _main+0x3): relocation GOT_LOAD is out of range: [[#]] is not in [-2147483648, 2147483647]; references _foo
# CHECK-DAG: error: {{.*}}test.o:(symbol _main+0x12): relocation BRANCH is out of range: -[[#]] is not in [-2147483648, 2147483647]{{$}}
# CHECK-DAG: error: stub is out of range: [[#]] is not in [-2147483648, 2147483647]; references _bar
# CHECK-DAG: error: stub helper header is out of range: [[#]] is not in [-2147483648, 2147483647]
# CHECK-DAG: error: stub helper header is out of range: [[#]] is not in [-2147483648, 2147483647]

## Also test that Writer::scanRelocations handles a subtractor minuend
## relocation with a null referent.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/subtractor-null-minuend.s -o %t/subtractor-null-minuend.o
# RUN: not %lld -lSystem -o /dev/null %t/subtractor-null-minuend.o 2>&1 | FileCheck %s --check-prefix=UNDEF
# UNDEF: error: undefined symbol: _main

#--- bar.s
.globl _bar
_bar:

#--- test.s

# Added to ensure that the error mentions the right function.
_baz:
  ret

.globl _main, _foo

_main:
  movq _foo@GOTPCREL(%rip), %rax
  callq _bar
  ret

.int _foo
  callq _ignored
.zerofill __TEXT,bss,_zero,0xffffffff

.data
_foo:
  .space 0

.section __DATA,__objc_imageinfo
_ignored:

#--- subtractor-null-minuend.s
_foo:
  .quad _ignored - _foo

.section __DATA,__objc_imageinfo
_ignored:
