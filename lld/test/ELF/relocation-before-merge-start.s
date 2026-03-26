// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
// RUN: not ld.lld %t.o -o /dev/null -shared 2>&1 | FileCheck %s --implicit-check-not=error:
// CHECK: error: {{.*}}:(.foo): offset is outside the section
// CHECK: error: {{.*}}:(.foo): offset is outside the section

.data
.quad .foo - 1
.quad .foo + 0x100000000
.section	.foo,"aM",@progbits,4
.quad 0
