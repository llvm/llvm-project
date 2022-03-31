# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: undefined symbol: undef

.global _start
_start:
  bl undef
  nop
