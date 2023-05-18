// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t --shared --icf=all -o %t.so
// The output file is large, most of it zeroes. We dissassemble only the
// parts we need to speed up the test and avoid a large output file
// RUN: llvm-objdump --no-print-imm-hex -d %t.so --start-address=0x2000000 --stop-address=0x2000018 | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.so --start-address=0x2800004 --stop-address=0x2800034 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.so --start-address=0x4000000 --stop-address=0x4000010 | FileCheck --check-prefix=CHECK3 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.so --start-address=0x4000010 --stop-address=0x4000100 --triple=armv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK4 %s
// RUN: rm %t.so
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
 .global far_nonpreemptible
 .hidden far_nonpreemptible
 .type far_nonpreemptible, %function
 .global far_nonpreemptible_alias
 .hidden far_nonpreemptible_alias
 .type far_nonpreemptible_alias, %function
sym1:
 bl elsewhere
 bl preemptible
 bx lr
preemptible:
 bl far_preemptible
 bl far_nonpreemptible
 bl far_nonpreemptible_alias
 bx lr
// CHECK1: Disassembly of section .text:
// CHECK1-EMPTY:
// CHECK1-NEXT: <sym1>:
// CHECK1-NEXT:  2000000:       f000 d800       bl      0x2800004 <__ThumbV7PILongThunk_elsewhere>
// CHECK1-NEXT:  2000004:       f000 d804       bl      0x2800010 <__ThumbV7PILongThunk_preemptible>
// CHECK1-NEXT:  2000008:       4770    bx      lr
// CHECK1: <preemptible>:
// CHECK1-NEXT:  200000a:       f000 d807       bl      0x280001c <__ThumbV7PILongThunk_far_preemptible>
// CHECK1-NEXT:  200000e:       f000 d80b       bl      0x2800028 <__ThumbV7PILongThunk_far_nonpreemptible>
// CHECK1-NEXT:  2000012:       f000 d809       bl      0x2800028 <__ThumbV7PILongThunk_far_nonpreemptible>
// CHECK1-NEXT:  2000016:       4770    bx      lr

 .section .text.2, "ax", %progbits
 .balign 0x0800000
 bx lr
// CHECK2: <__ThumbV7PILongThunk_elsewhere>:
// CHECK2-NEXT:  2800004:       f240 0c20       movw    r12, #32
// CHECK2-NEXT:  2800008:       f2c0 1c80       movt    r12, #384
// CHECK2-NEXT:  280000c:       44fc    add     r12, pc
// CHECK2-NEXT:  280000e:       4760    bx      r12
// CHECK2: <__ThumbV7PILongThunk_preemptible>:
// CHECK2-NEXT:  2800010:       f240 0c24       movw    r12, #36
// CHECK2-NEXT:  2800014:       f2c0 1c80       movt    r12, #384
// CHECK2-NEXT:  2800018:       44fc    add     r12, pc
// CHECK2-NEXT:  280001a:       4760    bx      r12
// CHECK2: <__ThumbV7PILongThunk_far_preemptible>:
// CHECK2-NEXT:  280001c:       f240 0c28       movw    r12, #40
// CHECK2-NEXT:  2800020:       f2c0 1c80       movt    r12, #384
// CHECK2-NEXT:  2800024:       44fc    add     r12, pc
// CHECK2-NEXT:  2800026:       4760    bx      r12
// CHECK2: <__ThumbV7PILongThunk_far_nonpreemptible>:
// CHECK2-NEXT:  2800028:       f64f 7ccd       movw    r12, #65485
// CHECK2-NEXT:  280002c:       f2c0 1c7f       movt    r12, #383
// CHECK2-NEXT:  2800030:       44fc    add     r12, pc
// CHECK2-NEXT:  2800032:       4760    bx      r12

 .section .text.3, "ax", %progbits
.balign 0x2000000
far_preemptible:
far_nonpreemptible:
 bl elsewhere

 .section .text.4, "ax", %progbits
.balign 0x2000000
far_nonpreemptible_alias:
 bl elsewhere

// CHECK3: <far_preemptible>:
// CHECK3:  4000000:       f000 e816       blx     0x4000030

// CHECK4: Disassembly of section .plt:
// CHECK4-EMPTY:
// CHECK4-NEXT: <.plt>:
// CHECK4-NEXT:  4000010:	e52de004    	str	lr, [sp, #-4]!
// CHECK4-NEXT:  4000014:	e28fe600    	add	lr, pc, #0, #12
// CHECK4-NEXT:  4000018:	e28eea20    	add	lr, lr, #32
// CHECK4-NEXT:  400001c:	e5bef0a4    	ldr	pc, [lr, #164]!
// CHECK4-NEXT:  4000020:	d4 d4 d4 d4 	.word	0xd4d4d4d4
// CHECK4-NEXT:  4000024:	d4 d4 d4 d4 	.word	0xd4d4d4d4
// CHECK4-NEXT:  4000028:	d4 d4 d4 d4 	.word	0xd4d4d4d4
// CHECK4-NEXT:  400002c:	d4 d4 d4 d4 	.word	0xd4d4d4d4
// CHECK4-NEXT:  4000030:	e28fc600    	add	r12, pc, #0, #12
// CHECK4-NEXT:  4000034:	e28cca20    	add	r12, r12, #32
// CHECK4-NEXT:  4000038:	e5bcf08c    	ldr	pc, [r12, #140]!
// CHECK4-NEXT:  400003c:	d4 d4 d4 d4 	.word	0xd4d4d4d4
// CHECK4-NEXT:  4000040:	e28fc600    	add	r12, pc, #0, #12
// CHECK4-NEXT:  4000044:	e28cca20    	add	r12, r12, #32
// CHECK4-NEXT:  4000048:	e5bcf080    	ldr	pc, [r12, #128]!
// CHECK4-NEXT:  400004c:	d4 d4 d4 d4 	.word	0xd4d4d4d4
// CHECK4-NEXT:  4000050:	e28fc600    	add	r12, pc, #0, #12
// CHECK4-NEXT:  4000054:	e28cca20    	add	r12, r12, #32
// CHECK4-NEXT:  4000058:	e5bcf074    	ldr	pc, [r12, #116]!
// CHECK4-NEXT:  400005c:	d4 d4 d4 d4 	.word	0xd4d4d4d4
