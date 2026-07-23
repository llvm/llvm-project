# REQUIRES: x86 && !thread_support
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o

# RUN: not %lld --read-workers=1 %t.o -o /dev/null

# CHECK: error: --read-workers=: option unavailable because lld was built without thread support

.globl _main
_main:
  ret
