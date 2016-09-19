# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %tout
# RUN: llvm-objdump --section-headers  %tout | FileCheck %s
# REQUIRES: x86

.global _start
.text
_start:

.section .text.a,"ax"
.byte 0
.section .text.,"ax"
.byte 0
.section .rodata.a,"a"
.byte 0
.section .rodata,"a"
.byte 0
.section .data.a,"aw"
.byte 0
.section .data,"aw"
.byte 0
.section .bss.a,"",@nobits
.byte 0
.section .bss,"",@nobits
.byte 0
.section .foo.a,"aw"
.byte 0
.section .foo,"aw"
.byte 0
.section .data.rel.ro,"aw",%progbits
.byte 0
.section .data.rel.ro.a,"aw",%progbits
.byte 0
.section .data.rel.ro.local,"aw",%progbits
.byte 0
.section .data.rel.ro.local.a,"aw",%progbits
.byte 0
.section .tbss.foo,"aGwT",@nobits,foo,comdat
.byte 0
.section .gcc_except_table.foo,"aG",@progbits,foo,comdat
.byte 0
.section .tdata.foo,"aGwT",@progbits,foo,comdat
.byte 0

// CHECK:  1 .rodata  00000002
// CHECK:  2 .gcc_except_table 00000001
// CHECK:  3 .text         00000002
// CHECK:  4 .tdata        00000001
// CHECK:  5 .tbss         00000001
// CHECK:  6 .data.rel.ro  00000004
// CHECK:  7 .data         00000002
// CHECK:  8 .foo.a        00000001
// CHECK:  9 .foo          00000001
// CHECK: 10 .bss          00000001
// CHECK: 11 .bss          00000001
// CHECK: 12 .symtab       00000060
// CHECK: 13 .shstrtab     0000006c
// CHECK: 14 .strtab       0000001d
