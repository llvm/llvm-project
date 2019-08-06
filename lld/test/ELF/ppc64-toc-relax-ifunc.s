# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: echo '.globl ifunc; .type ifunc, %gnu_indirect_function; ifunc:' | \
# RUN:   llvm-mc -filetype=obj -triple=powerpc64le - -o %t1.o
# RUN: ld.lld %t.o %t1.o -o %t
# RUN: llvm-objdump -d %t | FileCheck %s

## ifunc is a non-preemptable STT_GNU_IFUNC. Its toc entry will be
## relocated by R_PPC64_IRELATIVE, not representable by a toc-relative value.
## Check the toc-indirect access is not relaxed.

# CHECK:      nop
# CHECK-NEXT: ld 3, -32768(2)

addis 3, 2, .toc@toc@ha
ld 3, .toc@toc@l(3)

.section .toc,"aw",@progbits
  .quad ifunc
