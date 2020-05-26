# REQUIRES: ppc
## Test code sequences of synthesized _restgpr0_{14..31}

# RUN: llvm-mc -filetype=obj -triple=ppc64le %s -o %t14.o
# RUN: ld.lld %t14.o -o %t14
# RUN: llvm-objdump -d %t14 | FileCheck --check-prefix=R14 %s

# R14-LABEL: <_restgpr0_14>:
# R14-NEXT:    ld 14, -144(1)
# R14-NEXT:    ld 15, -136(1)
# R14-EMPTY:
# R14-NEXT:  <_restgpr0_16>:
# R14-NEXT:    ld 16, -128(1)
# R14:         ld 31, -8(1)
# R14-NEXT:    ld 0, 16(1)
# R14-NEXT:    mtlr 0
# R14-NEXT:    blr

## Don't synthesize _restgpr0_{14..30} because they are unused.
# RUN: echo 'bl _restgpr0_31' | llvm-mc -filetype=obj -triple=ppc64 - -o %t31.o
# RUN: ld.lld %t31.o -o %t31
# RUN: llvm-objdump -d %t31 | FileCheck --check-prefix=R31 %s

# R31-LABEL: Disassembly of section .text:
# R31-EMPTY:
# R31-NEXT:  <_restgpr0_31>:
# R31-NEXT:    ld 31, -8(1)
# R31-NEXT:    ld 0, 16(1)
# R31-NEXT:    mtlr 0
# R31-NEXT:    blr

# RUN: echo 'bl _restgpr0_32' | llvm-mc -filetype=obj -triple=ppc64 - -o %t32.o
# RUN: not ld.lld %t32.o -o /dev/null

.globl _start
_start:
  bl _restgpr0_14
  bl _restgpr0_16
