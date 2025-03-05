# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 -crel a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 -crel b.s -o b.o
# RUN: ld.lld -pie a.o b.o -o out
# RUN: llvm-objdump -d out | FileCheck %s
# RUN: llvm-readelf -Srs out | FileCheck %s --check-prefix=RELOC
# RUN: llvm-dwarfdump --eh-frame out | FileCheck %s --check-prefix=UNWIND

# CHECK:       <_start>:
# CHECK-NEXT:    callq {{.*}} <foo>
# CHECK-NEXT:    callq {{.*}} <foo>
# CHECK-EMPTY:
# CHECK-NEXT:  <foo>:
# CHECK-NEXT:    leaq {{.*}}  # 0x27c
# CHECK-NEXT:    leaq {{.*}}  # 0x278

# RELOC:  .data             PROGBITS        {{0*}}[[#%x,DATA:]]

# RELOC:  {{0*}}[[#DATA+8]]  0000000000000008 R_X86_64_RELATIVE [[#%x,DATA+0x8000000000000000]]

# RELOC:      00000000000012f4     0 NOTYPE  GLOBAL DEFAULT [[#]] _start
# RELOC-NEXT: 00000000000012fe     0 NOTYPE  GLOBAL DEFAULT [[#]] foo

## initial_location fields in FDEs are correctly relocated.
# UNWIND: 00000018 00000010 0000001c FDE cie=00000000 pc=000012f4...000012fe
# UNWIND: 0000002c 00000010 00000030 FDE cie=00000000 pc=000012fe...0000130c

# RUN: ld.lld -pie --emit-relocs a.o b.o -o out1
# RUN: llvm-objdump -dr out1 | FileCheck %s --check-prefix=CHECKE
# RUN: llvm-readelf -Sr out1 | FileCheck %s --check-prefix=RELOCE

# CHECKE:       <_start>:
# CHECKE-NEXT:    callq {{.*}} <foo>
# CHECKE-NEXT:      R_X86_64_PLT32 foo-0x4
# CHECKE-NEXT:    callq {{.*}} <foo>
# CHECKE-NEXT:      R_X86_64_PLT32 .text+0x6
# CHECKE-EMPTY:
# CHECKE-NEXT:  <foo>:
# CHECKE-NEXT:    leaq {{.*}}
# CHECKE-NEXT:      R_X86_64_PC32 .L.str-0x4
# CHECKE-NEXT:    leaq {{.*}}
# CHECKE-NEXT:      R_X86_64_PC32 .L.str1-0x4

# RELOCE:      .rodata             PROGBITS        {{0*}}[[#%x,RO:]]
# RELOCE:      .eh_frame           PROGBITS        {{0*}}[[#%x,EHFRAME:]]
# RELOCE:      .data               PROGBITS        {{0*}}[[#%x,DATA:]]

# RELOCE:      Relocation section '.crel.data' at offset {{.*}} contains 2 entries:
# RELOCE-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# RELOCE-NEXT: {{0*}}[[#DATA+8]] {{.*}}           R_X86_64_64            {{.*}}           .data - 8000000000000000
# RELOCE-NEXT: {{0*}}[[#DATA+24]]{{.*}}           R_X86_64_64            {{.*}}           .data - 1
# RELOCE:      Relocation section '.crel.eh_frame' at offset {{.*}} contains 2 entries:
# RELOCE-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# RELOCE-NEXT: {{0*}}[[#EHFRAME+32]] {{.*}}       R_X86_64_PC32          {{.*}}           .text + 0
# RELOCE-NEXT: {{0*}}[[#EHFRAME+52]] {{.*}}       R_X86_64_PC32          {{.*}}           .text + a
# RELOCE:      Relocation section '.crel.rodata' at offset {{.*}} contains 4 entries:
# RELOCE-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# RELOCE-NEXT: {{0*}}[[#RO+8]]   {{.*}}           R_X86_64_PC32          {{.*}}           foo + 0
# RELOCE-NEXT: {{0*}}[[#RO+23]]  {{.*}}           R_X86_64_PC32          {{.*}}           foo + 3f
# RELOCE-NEXT: {{0*}}[[#RO+39]]  {{.*}}           R_X86_64_PC64          {{.*}}           foo + 7f
# RELOCE-NEXT: {{0*}}[[#RO+47]]  {{.*}}           R_X86_64_PC32          {{.*}}           _start - 1f81

#--- a.s
.global _start, foo
_start:
  .cfi_startproc # Test .eh_frame
  call foo
  call .text.foo
  .cfi_endproc

.section .text.foo,"ax"
foo:
  .cfi_startproc
  leaq .L.str(%rip), %rsi
  leaq .L.str1(%rip), %rsi
  .cfi_endproc

.section .rodata.str1.1,"aMS",@progbits,1
.L.str:
  .asciz  "abc"
.L.str1:
  .asciz  "def"

.data
.quad 0
.quad .data - 0x8000000000000000
.quad 0
.quad .data - 1

#--- b.s
.section .rodata,"a"
.long foo - .
.space 15-4
.long foo - . + 63  # offset+=15
.space 16-4
.quad foo - . + 127  # offset+=16
.long _start - . - 8065
