# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld -pie %t.o -o %t.pie
# RUN: llvm-readobj -r --dyn-syms %t.pie | FileCheck %s

## Test that we create R_X86_64_RELATIVE relocations with -pie.
# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
# CHECK-NEXT:     0x3368 R_X86_64_RELATIVE - 0x3368
# CHECK-NEXT:     0x3370 R_X86_64_RELATIVE - 0x3370
# CHECK-NEXT:     0x3378 R_X86_64_RELATIVE - 0x3371
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
