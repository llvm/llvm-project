// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-none %s -o %t.o
// RUN: ld.lld %t.o -o %t.so -shared
// RUN: llvm-readelf -l %t.so | FileCheck --implicit-check-not=LOAD %s

// RUN: echo ".section .foo,\"ax\"; \
// RUN:       ret" > %t.s
// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-none %t.s -o %t2.o
// RUN: ld.lld %t.o %t2.o -o %t.so -shared
// RUN: llvm-readelf -l %t.so | FileCheck --check-prefix=DIFF --implicit-check-not=LOAD %s

// CHECK:      LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x000245 0x000245 R   0x10000
// CHECK:      LOAD           0x000248 0x0000000000010248 0x0000000000010248 0x{{.*}} 0x{{.*}} R E 0x10000
// CHECK:      LOAD           0x00024c 0x000000000002024c 0x000000000002024c 0x{{.*}} 0x{{.*}}   E 0x10000
// CHECK:      LOAD           0x000250 0x0000000000030250 0x0000000000030250 0x000070 0x000db0 RW  0x10000

// CHECK: 01     .dynsym .gnu.hash .hash .dynstr
// CHECK: 02     .text
// CHECK: 03     .foo
// CHECK: 04     .dynamic

// DIFF:      LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x00020d 0x00020d R   0x10000
// DIFF:      LOAD           0x000210 0x0000000000010210 0x0000000000010210 0x00000c 0x00000c R E 0x10000
// DIFF:      LOAD           0x000220 0x0000000000020220 0x0000000000020220 0x000070 0x000de0 RW  0x10000

// DIFF: 01     .dynsym .gnu.hash .hash .dynstr
// DIFF: 02     .text .foo
// DIFF: 03     .dynamic

        ret
        .section .foo,"axy"
        ret
