// REQUIRES: arm
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=armv7a-linux-gnueabi
// RUN: ld.lld %t.o -o %t.so -shared
// RUN: llvm-readobj -S --dyn-relocations %t.so | FileCheck --check-prefix=SEC %s
// RUN: llvm-objdump -d --triple=armv7a-linux-gnueabi %t.so | FileCheck %s

// Test the handling of the initial-exec TLS model. Relative location within
// static TLS is a run-time constant computed by dynamic loader as a result
// of the R_ARM_TLS_TPOFF32 relocation.

 .syntax unified
 .arm
 .globl func
 .type  func,%function
 .p2align 2
func:
.L0:
 nop
.L1:
 nop
.L2:
 nop

 .p2align 2
// Generate R_ARM_TLS_IE32 static relocations
// Allocates a GOT entry dynamically relocated by R_ARM_TLS_TPOFF32
// literal contains the offset of the GOT entry from the place
.Lt0: .word  x(gottpoff) + (. - .L0 - 8)
.Lt1: .word  y(gottpoff) + (. - .L1 - 8)
.Lt2: .word  .TLSSTART(gottpoff) + (. - .L2 - 8)

// __thread int x = 10
// __thread int y;
// __thread int z __attribute((visibility("hidden")))
 .hidden z
 .globl  z
 .globl  y
 .globl  x

 .section       .tbss,"awT",%nobits
 .p2align  2
.TLSSTART:
 .type  z, %object
z:
 .space 4
 .type  y, %object
y:
 .space 4
 .section       .tdata,"awT",%progbits
 .p2align 2
 .type  x, %object
x:
 .word  10

// SEC:      Name: .tdata
// SEC-NEXT: Type: SHT_PROGBITS
// SEC-NEXT: Flags [
// SEC-NEXT:   SHF_ALLOC
// SEC-NEXT:   SHF_TLS
// SEC-NEXT:   SHF_WRITE
// SEC:      Size: 4
// SEC:      Name: .tbss
// SEC-NEXT: Type: SHT_NOBITS
// SEC-NEXT: Flags [
// SEC-NEXT:   SHF_ALLOC
// SEC-NEXT:   SHF_TLS
// SEC-NEXT:   SHF_WRITE
// SEC: Size: 8

// SEC:      Name: .got
// SEC-NEXT: Type: SHT_PROGBITS
// SEC-NEXT: Flags [
// SEC-NEXT:    SHF_ALLOC
// SEC-NEXT:    SHF_WRITE
// SEC-NEXT: ]
// SEC-NEXT: Address: 0x2254
// SEC:      Size: 12


// SEC: Dynamic Relocations {
// SEC:  0x225C R_ARM_TLS_TPOFF32
// SEC:  0x2254 R_ARM_TLS_TPOFF32 x
// SEC:  0x2258 R_ARM_TLS_TPOFF32 y

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <func>:
// CHECK-NEXT:    11e8: 00 f0 20 e3     nop
// CHECK-NEXT:    11ec: 00 f0 20 e3     nop
// CHECK-NEXT:    11f0: 00 f0 20 e3     nop

// (0x2254 - 0x11f4) + (0x11f4 - 0x11e8 - 8) = 0x1064
// CHECK:         11f4: 64 10 00 00
// (0x2258 - 0x11f8) + (0x11f8 - 0x11ec - 8) = 0x1064
// CHECK-NEXT:    11f8: 64 10 00 00
// (0x225c - 0x11f8) + (0x11f8 - 0x11f0 - 8) = 0x1064
// CHECK-NEXT:    11fc: 64 10 00 00
