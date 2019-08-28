// REQUIRES: arm
// RUN: llvm-mc %s -o %t.o -filetype=obj -triple=armv7a-linux-gnueabi
// RUN: ld.lld %t.o -o %t.so -shared
// RUN: llvm-readobj -S --dyn-relocations %t.so | FileCheck --check-prefix=SEC %s
// RUN: llvm-objdump -d -triple=armv7a-linux-gnueabi %t.so | FileCheck %s
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d -triple=armv7a-linux-gnueabi %t | FileCheck --check-prefix=CHECK-EXE %s

// Test the handling of the local-dynamic TLS model. Dynamic loader finds
// module index R_ARM_TLS_DTPMOD32. The offset in the next GOT slot is 0
// The R_ARM_TLS_LDO is the offset of the variable within the TLS block.
 .global __tls_get_addr
 .text
 .p2align  2
 .global _start
 .syntax unified
 .arm
 .type   _start, %function
_start:
.L0:
 nop

 .word   x(tlsldm) + (. - .L0 - 8)
 .word   x(tlsldo)
 .word   y(tlsldo)

 .section        .tbss,"awT",%nobits
 .p2align  2
 .type   y, %object
y:
 .space  4
 .section        .tdata,"awT",%progbits
 .p2align  2
 .type   x, %object
x:
 .word   10

// SEC:      Name: .tdata
// SEC-NEXT: Type: SHT_PROGBITS
// SEC-NEXT: Flags [
// SEC-NEXT:   SHF_ALLOC
// SEC-NEXT:   SHF_TLS
// SEC-NEXT:   SHF_WRITE
// SEC-NEXT: ]
// SEC-NEXT: Address: 0x21D0
// SEC:    Size: 4
// SEC:    Name: .tbss
// SEC-NEXT: Type: SHT_NOBITS (0x8)
// SEC-NEXT: Flags [
// SEC-NEXT:   SHF_ALLOC
// SEC-NEXT:   SHF_TLS
// SEC-NEXT:   SHF_WRITE
// SEC-NEXT: ]
// SEC-NEXT: Address: 0x21D4
// SEC:      Size: 4

// SEC: Dynamic Relocations {
// SEC-NEXT:  0x2224 R_ARM_TLS_DTPMOD32 - 0x0

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: _start:
// CHECK-NEXT: 11c0:       00 f0 20 e3     nop

// (0x2224 - 0x11c4) + (0x11c4 - 0x11c0 - 8) = 0x105c
// CHECK:      11c4:       5c 10 00 00
// CHECK-NEXT: 11c8:       00 00 00 00
// CHECK-NEXT: 11cc:       04 00 00 00

// CHECK-EXE:      _start:
// CHECK-EXE-NEXT:   11114:       00 f0 20 e3     nop
// CHECK-EXE:        11118:       0c 10 00 00
// CHECK-EXE-NEXT:   1111c:       00 00 00 00
// CHECK-EXE-NEXT:   11120:       04 00 00 00
