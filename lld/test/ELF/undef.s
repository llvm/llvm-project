# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: not ld.lld %t -o %t2 2>&1 | FileCheck %s
# CHECK: Undefined symbol: bar in {{.*}}
# CHECK: Undefined symbol: foo in {{.*}}
# REQUIRES: x86

  .globl _start
_start:
  call foo
  call bar
