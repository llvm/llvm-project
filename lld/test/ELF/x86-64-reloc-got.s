// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld %t.o -shared -o %t.so
// RUN: llvm-readelf -s %t.so | FileCheck %s -check-prefix=SECTION
// RUN: llvm-objdump -d %t.so | FileCheck %s

// SECTION: .got PROGBITS 0000000000003070 003070 000000

// 0x3070 (.got end) - 0x1007 = 8297
// CHECK: gotpc32:
// CHECK-NEXT: 1000: {{.*}} leaq 8297(%rip), %r15
.global gotpc32
gotpc32:
  leaq _GLOBAL_OFFSET_TABLE_(%rip), %r15

// CHECK: gotpc64:
// CHECK-NEXT: 1007: {{.*}} movabsq $8297, %r11
.global gotpc64
gotpc64:
  movabsq $_GLOBAL_OFFSET_TABLE_-., %r11
