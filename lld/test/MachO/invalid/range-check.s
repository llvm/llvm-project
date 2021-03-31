# REQUIRES: x86

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: %lld -dylib %t/bar.o -o %t/libbar.dylib
# RUN: not %lld -lSystem -o /dev/null %t/libbar.dylib %t/test.o 2>&1 | FileCheck %s

# CHECK: error: relocation UNSIGNED is out of range: [[#]] is not in [0, 4294967295]; references _foo
# CHECK: error: relocation GOT_LOAD is out of range: [[#]] is not in [-2147483648, 2147483647]; references _foo
# CHECK: error: stub is out of range: [[#]] is not in [-2147483648, 2147483647]; references _bar
# CHECK: error: stub helper header is out of range: [[#]] is not in [-2147483648, 2147483647]
# CHECK: error: stub helper header is out of range: [[#]] is not in [-2147483648, 2147483647]

#--- bar.s
.globl _bar
_bar:

#--- test.s
.globl _main, _foo

_main:
  movq _foo@GOTPCREL(%rip), %rax
  callq _bar
  ret

.int _foo
.zerofill __TEXT,bss,_zero,0xffffffff

.data
_foo:
  .space 0
