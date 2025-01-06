# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %tout
# RUN: llvm-objdump --section-headers  %tout | FileCheck %s

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
.section .bss.a,"aw",@nobits
.byte 0
.section .bss,"aw",@nobits
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
.section .sdata,"aw"
.byte 0
.section .sdata.foo,"aw"
.byte 0
.section .sbss,"aw",@nobits
.byte 0
.section .sbss.foo,"aw",@nobits
.byte 0
.section .srodata,"a"
.byte 0
.section .srodata.foo,"a"
.byte 0

// CHECK:      .rodata           00000002
// CHECK-NEXT: .gcc_except_table 00000001
// CHECK-NEXT: .srodata          00000002
// CHECK-NEXT: .text             00000002
// CHECK-NEXT: .tdata            00000001
// CHECK-NEXT: .tbss             00000001
// CHECK-NEXT: .data.rel.ro      00000004
// CHECK-NEXT: .relro_padding    00000df1
// CHECK-NEXT: .data             00000002
// CHECK-NEXT: .foo.a            00000001
// CHECK-NEXT: .foo              00000001
// CHECK-NEXT: .sdata            00000002
// CHECK-NEXT: .bss              00000002
// CHECK-NEXT: .sbss             00000002
// CHECK-NEXT: .comment          00000008
// CHECK-NEXT: .symtab           00000030
// CHECK-NEXT: .shstrtab         0000009a
// CHECK-NEXT: .strtab           00000008
