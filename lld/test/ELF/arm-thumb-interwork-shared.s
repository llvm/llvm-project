// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t --shared -o %t.so
// RUN: llvm-objdump -d -triple=thumbv7a-none-linux-gnueabi %t.so | FileCheck %s
 .syntax unified
 .global sym1
 .global elsewhere
 .weak weakref
sym1:
 b.w elsewhere
 b.w weakref

// Check that we generate a thunk for an undefined symbol called via a plt
// entry.

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: sym1:
// CHECK-NEXT: 11e0: 00 f0 02 b8 b.w #4 <__ThumbV7PILongThunk_elsewhere>
// CHECK-NEXT: 11e4: 00 f0 06 b8 b.w #12 <__ThumbV7PILongThunk_weakref>
// CHECK: __ThumbV7PILongThunk_elsewhere:
// CHECK-NEXT:     11e8:       40 f2 2c 0c     movw    r12, #44
// CHECK-NEXT:     11ec:       c0 f2 00 0c     movt    r12, #0
// CHECK-NEXT:     11f0:       fc 44   add     r12, pc
// CHECK-NEXT:     11f2:       60 47   bx      r12
// CHECK: __ThumbV7PILongThunk_weakref:
// CHECK-NEXT:     11f4:       40 f2 30 0c     movw    r12, #48
// CHECK-NEXT:     11f8:       c0 f2 00 0c     movt    r12, #0
// CHECK-NEXT:     11fc:       fc 44   add     r12, pc
// CHECK-NEXT:     11fe:       60 47   bx      r12

// CHECK: Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: $a:
// CHECK-NEXT:     1200:  04 e0 2d e5     str     lr, [sp, #-4]!
// CHECK-NEXT:     1204:  00 e6 8f e2     add     lr, pc, #0, #12
// CHECK-NEXT:     1208:  02 ea 8e e2     add     lr, lr, #8192
// CHECK-NEXT:     120c:  94 f0 be e5     ldr     pc, [lr, #148]!
// CHECK: $d:
// CHECK-NEXT:     1210:  d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-NEXT:     1214:  d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-NEXT:     1218:  d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-NEXT:     121c:  d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK: $a:
// CHECK-NEXT:     1220:  00 c6 8f e2     add     r12, pc, #0, #12
// CHECK-NEXT:     1224:  02 ca 8c e2     add     r12, r12, #8192
// CHECK-NEXT:     1228:  7c f0 bc e5     ldr     pc, [r12, #124]!
// CHECK: $d:
// CHECK-NEXT:     122c:  d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK: $a:
// CHECK-NEXT:     1230:  00 c6 8f e2     add     r12, pc, #0, #12
// CHECK-NEXT:     1234:  02 ca 8c e2     add     r12, r12, #8192
// CHECK-NEXT:     1238:  70 f0 bc e5     ldr     pc, [r12, #112]!
// CHECK: $d:
// CHECK-NEXT:     123c:  d4 d4 d4 d4     .word   0xd4d4d4d4
