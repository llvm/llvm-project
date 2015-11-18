# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %tout
# RUN: llvm-readobj -sections %tout | FileCheck %s
# REQUIRES: x86

.global _start
.text
_start:

.section .text.a,"ax"
.section .text.,"ax"
.section .rodata.a,"a"
.section .rodata,"a"
.section .data.a,"aw"
.section .data,"aw"
.section .bss.a,"",@nobits
.section .bss,"",@nobits
.section .foo.a,"aw"
.section .foo,"aw"
.section .data.rel.ro,"aw",%progbits
.section .data.rel.ro.a,"aw",%progbits
.section .data.rel.ro.local,"aw",%progbits
.section .data.rel.ro.local.a,"aw",%progbits

// CHECK-NOT: Name: .rodata.a
// CHECK:     Name: .rodata
// CHECK-NOT: Name: .text.a
// CHECK:     Name: .text
// CHECK-NOT: Name: .data.a
// CHECK:     Name: .data
// CHECK:     Name: .foo.a
// CHECK:     Name: .foo
// CHECK-NOT: Name: .data.rel.ro.a
// CHECK-NOT: Name: .data.rel.ro.local.a
// CHECK:     Name: .data.rel.ro
// CHECK-NOT: Name: .bss.a
// CHECK:     Name: .bss
