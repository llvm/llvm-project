# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld -pie %t.o -o %t.pie
# RUN: llvm-readobj -r --dyn-syms %t.pie | FileCheck %s

## Test that we create R_X86_64_RELATIVE relocations with -pie.
# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
# CHECK-NEXT:     0x[[FOO_ADDR:.*]] R_X86_64_RELATIVE - 0x[[FOO_ADDR]]
# CHECK-NEXT:     0x[[#%X,BAR_ADDR:]] R_X86_64_RELATIVE
# CHECK-SAME:       - 0x[[#BAR_ADDR]]
# CHECK-NEXT:     0x[[#BAR_ADDR + 8]] R_X86_64_RELATIVE - 0x[[#BAR_ADDR + 1]]
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.globl _start
_start:
nop

 .data
foo:
 .quad foo

.hidden bar
.global bar
bar:
 .quad bar
 .quad bar + 1
