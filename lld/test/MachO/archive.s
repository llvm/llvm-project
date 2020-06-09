# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: echo ".global _boo; _boo: ret"                           | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/2.o
# RUN: echo ".global _bar; _bar: ret"                           | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/3.o
# RUN: echo ".global _undefined; .global _unused; _unused: ret" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/4.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/main.o

# RUN: rm -f %t/test.a
# RUN: llvm-ar rcs %t/test.a %t/2.o %t/3.o %t/4.o
# RUN: lld -flavor darwinnew -arch x86_64 %t/main.o %t/test.a -o %t/test.out

## TODO: Run llvm-nm -p to validate symbol order
# RUN: llvm-nm %t/test.out | FileCheck %s
# CHECK: T _bar
# CHECK: T _boo
# CHECK: T _main

## Linking with the archive first in the command line shouldn't change anything
# RUN: lld -flavor darwinnew -arch x86_64 %t/test.a %t/main.o -o %t/test.out
# RUN: llvm-nm %t/test.out | FileCheck %s --check-prefix ARCHIVE-FIRST
# ARCHIVE-FIRST: T _bar
# ARCHIVE-FIRST: T _boo
# ARCHIVE-FIRST: T _main


# RUN: llvm-nm %t/test.out | FileCheck %s --check-prefix VISIBLE
# VISIBLE-NOT: T _undefined
# VISIBLE-NOT: T _unused

.global _main
_main:
  callq _boo
  callq _bar
  mov $0, %rax
  ret
