// REQUIRES: arm

// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %p/Inputs/arm-shared.s -o %t.so.o
// RUN: ld.lld -shared %t.so.o -soname=t.so -o %t.so

// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld -pie --pack-dyn-relocs=relr %t.o %t.so -o %t.exe
// RUN: llvm-readobj -r %t.exe | FileCheck %s

// CHECK:      Section (5) .relr.dyn {
// CHECK-NEXT:   0x301E8 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x301EC R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x301F0 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x301F4 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x301F8 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x301FC R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30200 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30204 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30208 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3020C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30210 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30214 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30218 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3021C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30220 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30224 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30228 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3022C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30230 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30234 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30238 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3023C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30240 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30244 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30248 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3024C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30250 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30254 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30258 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3025C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30260 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30264 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x30268 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3026C R_ARM_RELATIVE - 0x0
// CHECK-NEXT: }

// RUN: llvm-readobj -S --dynamic-table %t.exe | FileCheck --check-prefix=HEADER %s
// HEADER: 0x00000023 RELRSZ 0xC

.data
.align 2
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
.dc.a __ehdr_start
