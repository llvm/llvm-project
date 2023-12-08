// Test that exidx output sections are created correctly for each partition.

// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -o %t -shared --gc-sections

// RUN: llvm-objcopy --extract-main-partition %t %t0
// RUN: llvm-objcopy --extract-partition=part1 %t %t1

// Change upper case to lower case so that we can match unwind info (which is dumped
// in upper case) against program headers (which are dumped in lower case).
// RUN: llvm-readelf -l --unwind %t0 | tr A-Z a-z | FileCheck --ignore-case %s
// RUN: llvm-readelf -l --unwind %t1 | tr A-Z a-z | FileCheck --ignore-case %s

// CHECK: LOAD  {{[^ ]*}} 0x{{0*}}[[TEXT_ADDR:[0-9a-f]+]] {{.*}} R E
// CHECK: EXIDX 0x{{0*}}[[EXIDX_OFFSET:[0-9a-f]+]] {{.*}} 0x00010 0x00010 R

// Each file should have one exidx section for its text section and one sentinel.
// CHECK:      SectionOffset: 0x[[EXIDX_OFFSET]]
// CHECK-NEXT: Entries [
// CHECK-NEXT:   Entry {
// CHECK-NEXT:     Functionaddress: 0x[[TEXT_ADDR]]
// CHECK-NEXT:     Model: CantUnwind
// CHECK-NEXT:   }
// CHECK-NEXT:   Entry {
// CHECK-NEXT:     FunctionAddress:
// CHECK-NEXT:     Model: CantUnwind
// CHECK-NEXT:   }
// CHECK-NEXT: ]

.section .llvm_sympart,"",%llvm_sympart
.asciz "part1"
.4byte p1

.section .text.p0,"ax",%progbits
.globl p0
p0:
.fnstart
bx lr
.cantunwind
.fnend

.section .text.p1,"ax",%progbits
.globl p1
p1:
.fnstart
bx lr
.cantunwind
.fnend
