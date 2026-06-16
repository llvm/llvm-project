# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: not ld.lld %t.o --irpgo-profile=%t.missing.profdata --bp-startup-sort=function 2>&1 | FileCheck -DMSG=%errc_ENOENT %s --check-prefix=MISSING-PROFILE

# MISSING-PROFILE: error: [[MSG]]

.globl _start
_start:
  ret
