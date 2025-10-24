# REQUIRES: x86 && thread_support
## Sometimes fails, particularly in an ASAN build, do not run until
## https://github.com/llvm/llvm-project/pull/157917 addresses the cause.
# UNSUPPORTED: target={{.*}}
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

## A non-negative integer is allowed.
# RUN: %lld --read-workers=0 %t.o -o /dev/null
# RUN: %lld --read-workers=1 %t.o -o /dev/null
# RUN: %lld --read-workers=2 %t.o -o /dev/null

# RUN: not %lld --read-workers=all %t.o -o /dev/null 2>&1 | FileCheck %s -DN=all
# RUN: not %lld --read-workers=-1 %t.o -o /dev/null 2>&1 | FileCheck %s -DN=-1

# CHECK: error: --read-workers=: expected a non-negative integer, but got '[[N]]'

.globl _main
_main:
  ret
