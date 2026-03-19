## Check skip-inline flag behavior

# RUN: llvm-mc --filetype=obj --triple=x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe --inline-small-functions --print-finalized --print-only=main \
# RUN:   -o %t.null | FileCheck %s --check-prefix=CHECK-INLINE
# RUN: llvm-bolt %t.exe --inline-small-functions --skip-inline=foo --print-finalized \
# RUN:   --print-only=main -o %t.null | FileCheck %s --check-prefix=CHECK-NO-INLINE
# CHECK-INLINE: Binary Function "main"
# CHECK-INLINE: ud2
# CHECK-NO-INLINE: Binary Function "main"
# CHECK-NO-INLINE: callq foo

.globl _start
_start:
  call main

.globl main
main:
  call foo
  ret

.globl foo
foo:
  ud2
  ret
