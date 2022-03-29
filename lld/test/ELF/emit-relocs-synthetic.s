# REQUIRES: x86
## Regression test: add STT_SECTION even if synthetic sections are interleaved
## with regular input sections.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: ld.lld --emit-relocs --no-relax -T %t/1.t %t/a.o -o %t/a1
# RUN: llvm-objdump -dr %t/a1 | FileCheck %s --check-prefixes=CHECK,CHECK1
# RUN: ld.lld --emit-relocs --no-relax -T %t/2.t %t/a.o -o %t/a2
# RUN: llvm-objdump -dr %t/a2 | FileCheck %s --check-prefixes=CHECK,CHECK2

# CHECK:       <_start>:
## %t/a1: bss is at offset 17. bss-4 = .bss + 0xd
## %t/a2: bss is at offset 16. bss-4 = .bss + 0xc
# CHECK-NEXT:    movl [[#]](%rip), %eax
# CHECK1-NEXT:     R_X86_64_PC32        .bss+0xd
# CHECK2-NEXT:     R_X86_64_PC32        .bss+0xc
# CHECK-NEXT:    movl [[#]](%rip), %eax
# CHECK-NEXT:      R_X86_64_PC32        common-0x4
# CHECK-NEXT:    movl [[#]](%rip), %eax
## %t/a1: input .data is at offset 8. 8-4 = 0x4
## %t/a2: input .data is at offset 0. 0-4 = -0x4
# CHECK1-NEXT:     R_X86_64_GOTPCRELX   .data+0x4
# CHECK2-NEXT:     R_X86_64_GOTPCRELX   .data-0x4

#--- a.s
.globl _start
_start:
  movl bss(%rip), %eax
  movl common(%rip), %eax
## Compilers don't produce this. We just check the behavior.
  movl .data@gotpcrel(%rip), %eax

.section .data,"aw",@progbits
.quad 0

.section .bss,"aw",@nobits
.space 16
bss:
.byte 0

.comm common,1,1

#--- 1.t
SECTIONS {
  .data : { *(.got) *(.data) }
  .bss : { *(COMMON) *(.bss) }
}

#--- 2.t
SECTIONS {
  .data : { *(.data) *(.got) }
  .bss : { *(.bss) *(COMMON) }
}
