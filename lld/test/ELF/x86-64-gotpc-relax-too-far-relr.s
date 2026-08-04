# REQUIRES: x86
## Test that RELR relocations added when unrelaxing GOTPCREL relocations
## during layout optimization (relaxOnce) are included in .relr.dyn.
## .quad foo ensures .relr.dyn is not removed before unrelaxing.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -pie --pack-dyn-relocs=relr --section-start=.text=0x10000 --section-start=.got=0x20000 --section-start=.data=0x100000000 %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .relr.dyn {
# CHECK-NEXT:     0x20000 R_X86_64_RELATIVE -
# CHECK-NEXT:     0x100000000 R_X86_64_RELATIVE -
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.text
.globl _start
_start:
  movq foo@GOTPCREL(%rip), %rax

.section .data,"aw",@progbits
.align 8
.globl foo
foo:
  .quad foo
