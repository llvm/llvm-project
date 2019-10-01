// REQUIRES: arm

// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %p/Inputs/arm-shared.s -o %t.so.o
// RUN: ld.lld -shared %t.so.o -soname=t.so -o %t.so

// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld -pie --pack-dyn-relocs=relr %t.o %t.so -o %t.exe
// RUN: llvm-readobj -r %t.exe | FileCheck %s

// CHECK:      Section (5) .relr.dyn {
// CHECK-NEXT:   0x31E0 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x31E4 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x31E8 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x31EC R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x31F0 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x31F4 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x31F8 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x31FC R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3200 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3204 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3208 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x320C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3210 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3214 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3218 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x321C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3220 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3224 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3228 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x322C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3230 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3234 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3238 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x323C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3240 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3244 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3248 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x324C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3250 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3254 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3258 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x325C R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3260 R_ARM_RELATIVE - 0x0
// CHECK-NEXT:   0x3264 R_ARM_RELATIVE - 0x0
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
