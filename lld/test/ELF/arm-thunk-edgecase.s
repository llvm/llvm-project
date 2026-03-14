// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:           .text_armfunc 0x1000 : { *(.text_armfunc) } \
// RUN:           .text_thumbfunc 0x11010 : { *(.text_thumbfunc) } \
// RUN:       }" > %tarm_to_thumb.script
// RUN: echo "SECTIONS { \
// RUN:           .text_thumbfunc 0x1000 : { *(.text_thumbfunc) } \
// RUN:           .text_armfunc 0x1100c : { *(.text_armfunc) } \
// RUN:       }" > %tthumb_to_arm.script
// RUN: ld.lld -shared -Bsymbolic -script %tarm_to_thumb.script %t.o -o %tarm_to_thumb.so
// RUN: ld.lld -shared -Bsymbolic -script %tthumb_to_arm.script %t.o -o %tthumb_to_arm.so
// RUN: llvm-objdump --no-print-imm-hex --triple=armv7a-none-linux-gnueabi -d %tarm_to_thumb.so | FileCheck --check-prefix=ARM-TO-THUMB %s
// RUN: llvm-objdump --no-print-imm-hex -d %tthumb_to_arm.so | FileCheck --check-prefix=THUMB-TO-ARM %s

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7aeb-none-linux-gnueabi -mcpu=cortex-a8 %s -o %t.o
// RUN: ld.lld -shared -Bsymbolic -script %tarm_to_thumb.script %t.o -o %tarm_to_thumb.so
// RUN: ld.lld -shared -Bsymbolic -script %tthumb_to_arm.script %t.o -o %tthumb_to_arm.so
// RUN: llvm-objdump --no-print-imm-hex --triple=armv7aeb-none-linux-gnueabi -d %tarm_to_thumb.so | FileCheck --check-prefix=ARM-TO-THUMB %s
// RUN: llvm-objdump --no-print-imm-hex -d %tthumb_to_arm.so | FileCheck --check-prefix=THUMB-TO-ARM %s

// RUN: ld.lld --be8 -shared -Bsymbolic -script %tarm_to_thumb.script %t.o -o %tarm_to_thumb.so
// RUN: ld.lld --be8 -shared -Bsymbolic -script %tthumb_to_arm.script %t.o -o %tthumb_to_arm.so
// RUN: llvm-objdump --no-print-imm-hex --triple=armv7aeb-none-linux-gnueabi -d %tarm_to_thumb.so | FileCheck --check-prefix=ARM-TO-THUMB %s
// RUN: llvm-objdump --no-print-imm-hex -d %tthumb_to_arm.so | FileCheck --check-prefix=THUMB-TO-ARM %s

.syntax unified

.arm
.section .text_armfunc, "ax", %progbits
.globl armfunc
.type armfunc, %function
armfunc:
	b	thumbfunc

.thumb
.section .text_thumbfunc, "ax", %progbits
.globl thumbfunc
.thumb_func
thumbfunc:
	b.w	armfunc

// ARM-TO-THUMB:      <__ARMV7PILongThunk_thumbfunc>:
// ARM-TO-THUMB-NEXT:     1004:        e30fcffd            movw        r12, #65533
// ARM-TO-THUMB-NEXT:     1008:        e340c000            movt        r12, #0

// THUMB-TO-ARM:      <__ThumbV7PILongThunk_armfunc>:
// THUMB-TO-ARM-NEXT:     1004:        f64f 7cfc           movw        r12, #65532
// THUMB-TO-ARM-NEXT:     1008:        f2c0 0c00           movt        r12, #0
