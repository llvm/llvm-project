# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: a section .foo with SHF_LINK_ORDER should not refer a non-regular section: {{.*}}.o:(.merge)

.section .merge,"aM",@progbits,8
.quad 0
.section .foo,"ao",@progbits,.merge
.quad 0
