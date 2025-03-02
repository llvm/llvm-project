// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s
// RUN: ld.lld %t.o -o /dev/null -shared --noinhibit-exec 2>&1 | FileCheck %s
// CHECK: .o:(.foo): offset is outside the section
// CHECCK: .o:(.rodata.str1.1): offset is outside the section

.data
.quad .foo + 10
.quad .rodata.str1.1 + 4

.section	.foo,"aM",@progbits,4
.quad 0

.section	.rodata.str1.1,"aMS",@progbits,1
.asciz	"abc"
