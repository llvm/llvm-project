# REQUIRES: x86
## A SHF_LINK_ORDER section linked to an SHF_EXCLUDE'd section is discarded
## along with it in a final link, and kept, as SHF_EXCLUDE itself is, in a
## relocatable link. GNU ld does the same.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 excl.s -o excl.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 main.s -o main.o

# RUN: ld.lld main.o excl.o -o out
# RUN: llvm-readelf -S out | FileCheck %s \
# RUN:     --implicit-check-not='{{ }}__patchable_function_entries ' \
# RUN:     --implicit-check-not='{{ }}.text.X '
# CHECK: {{ }}.text PROGBITS

# RUN: ld.lld -r main.o excl.o -o out.ro
# RUN: llvm-readelf -S out.ro | FileCheck %s --check-prefix=REL
# REL: [[#%u,X:]]] .text.X PROGBITS {{.*}} AXE
# REL: {{ }}__patchable_function_entries PROGBITS {{.*}} WAL [[#%u,X]]

#--- excl.s
.section .text.X,"axe",@progbits
X:
  retq

.section __patchable_function_entries,"awo",@progbits,.text.X
  .quad X

#--- main.s
.globl _start
_start:
  retq
