// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t --shared -o %t.so
// The output file is large, most of it zeroes. We dissassemble only the
// parts we need to speed up the test and avoid a large output file
// RUN: llvm-objdump -d %t.so -start-address=8388608 -stop-address=8388624 -triple=thumbv7a-linux-gnueabihf | FileCheck -check-prefix=CHECK1 %s
// RUN: llvm-objdump -d %t.so -start-address=16777216 -stop-address=16777256 -triple=thumbv7a-linux-gnueabihf | FileCheck -check-prefix=CHECK2 %s
// RUN: llvm-objdump -d %t.so -start-address=25165824 -stop-address=25165828 -triple=thumbv7a-linux-gnueabihf | FileCheck -check-prefix=CHECK3 %s
// RUN: llvm-objdump -d %t.so -start-address=25165828 -stop-address=25165908 -triple=armv7a-linux-gnueabihf | FileCheck -check-prefix=CHECK4 %s
 .syntax unified
 .thumb

// Make sure that we generate a range extension thunk to a PLT entry
 .section ".text.1", "ax", %progbits
 .global sym1
 .global elsewhere
 .type elsewhere, %function
 .global preemptible
 .type preemptible, %function
 .global far_preemptible
 .type far_preemptible, %function
sym1:
 bl elsewhere
 bl preemptible
 bx lr
preemptible:
 bl far_preemptible
 bx lr
// CHECK1: Disassembly of section .text:
// CHECK1-NEXT: sym1:
// CHECK1-NEXT:   800000:       00 f0 00 d8     bl      #8388608
// CHECK1-NEXT:   800004:       00 f0 04 d8     bl      #8388616
// CHECK1-NEXT:   800008:       70 47   bx      lr
// CHECK1: preemptible:
// CHECK1-NEXT:   80000a:       00 f0 07 d8     bl      #8388622
// CHECK1-NEXT:   80000e:       70 47   bx      lr

 .section .text.2, "ax", %progbits
 .balign 0x0800000
 bx lr
// CHECK2: __ThumbV7PILongThunk_elsewhere:
// CHECK2-NEXT:  1000004:       40 f2 14 0c     movw    r12, #20
// CHECK2-NEXT:  1000008:       c0 f2 80 0c     movt    r12, #128
// CHECK2-NEXT:  100000c:       fc 44   add     r12, pc
// CHECK2-NEXT:  100000e:       60 47   bx      r12
// CHECK2: __ThumbV7PILongThunk_preemptible:
// CHECK2-NEXT:  1000010:       40 f2 18 0c     movw    r12, #24
// CHECK2-NEXT:  1000014:       c0 f2 80 0c     movt    r12, #128
// CHECK2-NEXT:  1000018:       fc 44   add     r12, pc
// CHECK2-NEXT:  100001a:       60 47   bx      r12
// CHECK2: __ThumbV7PILongThunk_far_preemptible:
// CHECK2-NEXT:  100001c:       40 f2 1c 0c     movw    r12, #28
// CHECK2-NEXT:  1000020:       c0 f2 80 0c     movt    r12, #128
// CHECK2-NEXT:  1000024:       fc 44   add     r12, pc
// CHECK2-NEXT:  1000026:       60 47   bx      r12
 .section .text.3, "ax", %progbits
.balign 0x0800000
far_preemptible:
 bl elsewhere
// CHECK3: far_preemptible:
// CHECK3:  1800000:       00 f0 10 e8     blx     #32

// CHECK4: Disassembly of section .plt:
// CHECK4-NEXT: $a:
// CHECK4-NEXT:  1800010:       04 e0 2d e5     str     lr, [sp, #-4]!
// CHECK4-NEXT:  1800014:       04 e0 9f e5     ldr     lr, [pc, #4]
// CHECK4-NEXT:  1800018:       0e e0 8f e0     add     lr, pc, lr
// CHECK4-NEXT:  180001c:       08 f0 be e5     ldr     pc, [lr, #8]!
// CHECK4: $d:
// CHECK4-NEXT:  1800020:       e0 0f 00 00     .word   0x00000fe0
// CHECK4: $a:
// CHECK4-NEXT:  1800024:       04 c0 9f e5     ldr     r12, [pc, #4]
// CHECK4-NEXT:  1800028:       0f c0 8c e0     add     r12, r12, pc
// CHECK4-NEXT:  180002c:       00 f0 9c e5     ldr     pc, [r12]
// CHECK4: $d:
// CHECK4-NEXT:  1800030:       dc 0f 00 00     .word   0x00000fdc
// CHECK4: $a:
// CHECK4-NEXT:  1800034:       04 c0 9f e5     ldr     r12, [pc, #4]
// CHECK4-NEXT:  1800038:       0f c0 8c e0     add     r12, r12, pc
// CHECK4-NEXT:  180003c:       00 f0 9c e5     ldr     pc, [r12]
// CHECK4: $d:
// CHECK4-NEXT:  1800040:       d0 0f 00 00     .word   0x00000fd0
// CHECK4: $a:
// CHECK4-NEXT:  1800044:       04 c0 9f e5     ldr     r12, [pc, #4]
// CHECK4-NEXT:  1800048:       0f c0 8c e0     add     r12, r12, pc
// CHECK4-NEXT:  180004c:       00 f0 9c e5     ldr     pc, [r12]
// CHECK4: $d:
// CHECK4-NEXT:  1800050:       c4 0f 00 00     .word   0x00000fc4
