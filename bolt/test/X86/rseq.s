## Check that llvm-bolt avoids optimization of functions referenced from
## __rseq_cs section, i.e. containing critical sections used by restartable
## sequences in tcmalloc.

# RUN: %clang %cflags %s -o %t -nostdlib -no-pie -Wl,-q
# RUN: llvm-bolt %t -o %t.bolt --print-cfg 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIE
# RUN: %clang %cflags %s -o %t.pie -nostdlib -pie -Wl,-q
# RUN: llvm-bolt %t.pie -o %t.pie.bolt 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-PIE

# CHECK-NO-PIE: Binary Function "_start"
# CHECK-NO-PIE: IsSimple
# CHECK-NO-PIE-SAME: 0

# CHECK-PIE: restartable sequence detected in _start

.global _start
  .type _start, %function
_start:
        pushq %rbp
        mov %rsp, %rbp
.L1:
        pop %rbp
        retq
.size _start, .-_start

.reloc 0, R_X86_64_NONE

.section __rseq_cs, "aw"
.balign 32
  .quad  .L1
