# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/2.s -o %t/2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/3.s -o %t/3.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/4.s -o %t/4.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o

# RUN: llvm-ar rcs %t/test.a %t/2.o %t/3.o %t/4.o
# RUN: %lld %t/main.o %t/test.a -o %t/test.out

## TODO: Run llvm-nm -p to validate symbol order
# RUN: llvm-nm %t/test.out | FileCheck %s
# CHECK: T _bar
# CHECK: T _boo
# CHECK: T _main

## Linking with the archive first in the command line shouldn't change anything
# RUN: %lld %t/test.a %t/main.o -o %t/test.out
# RUN: llvm-nm %t/test.out | FileCheck %s --check-prefix ARCHIVE-FIRST
# ARCHIVE-FIRST: T _bar
# ARCHIVE-FIRST: T _boo
# ARCHIVE-FIRST: T _main

# RUN: llvm-nm %t/test.out | FileCheck %s --check-prefix VISIBLE
# VISIBLE-NOT: T _undefined
# VISIBLE-NOT: T _unused

# RUN: %lld %t/test.a %t/main.o -o %t/all-load -noall_load -all_load
# RUN: llvm-nm %t/all-load | FileCheck %s --check-prefix ALL-LOAD
# ALL-LOAD: T _bar
# ALL-LOAD: T _boo
# ALL-LOAD: T _main
# ALL-LOAD: T _unused

# RUN: %lld %t/test.a %t/main.o -o %t/no-all-load -all_load -noall_load
# RUN: llvm-nm %t/no-all-load | FileCheck %s --check-prefix NO-ALL-LOAD
# RUN: %lld %t/test.a %t/main.o -o %t/no-all-load-only -noall_load
# RUN: llvm-nm %t/no-all-load-only | FileCheck %s --check-prefix NO-ALL-LOAD
# NO-ALL-LOAD-NOT: T _unused

## Multiple archives defining the same symbols aren't an issue, due to lazy
## loading
# RUN: cp %t/test.a %t/test2.a
# RUN: %lld %t/test.a %t/test2.a %t/main.o -o /dev/null

#--- 2.s
.globl _boo
_boo:
  ret

#--- 3.s
.globl _bar
_bar:
  ret

#--- 4.s
.globl _undefined, _unused
_unused:
  ret

#--- main.s
.globl _main
_main:
  callq _boo
  callq _bar
  ret
