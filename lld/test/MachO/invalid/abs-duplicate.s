# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weakfoo.s -o %t/weakfoo.o
# RUN: not %lld -lSystem %t/test.o %t/weakfoo.o -o %t/test 2>&1 | FileCheck %s

# CHECK: lld: error: duplicate symbol: _weakfoo

#--- weakfoo.s
.globl _weakfoo
## The weak attribute is ignored for absolute symbols, so we will have a
## duplicate symbol error for _weakfoo.
.weak_definition _weakfoo
_weakfoo = 0x1234

#--- test.s
.globl _main, _weakfoo
.weak_definition _weakfoo
_weakfoo = 0x5678

.text
_main:
  ret
