# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: not ld.lld2 --no-undefined -shared %t -o %t.so
# RUN: ld.lld2 -shared %t -o %t1.so

.globl _shared
_shared:
  call _unresolved
