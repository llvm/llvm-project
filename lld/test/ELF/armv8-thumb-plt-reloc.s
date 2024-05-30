// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -arm-add-build-attributes -triple=thumbv8-none-linux-gnueabi --arch=thumb --mcpu=cortex-m33 %p/Inputs/arm-plt-reloc.s -o %t1.o
// RUN: llvm-mc -filetype=obj -arm-add-build-attributes -triple=thumbv8-none-linux-gnueabi --arch=thumb --mcpu=cortex-m33 %s -o %t2.o
// RUN: ld.lld %t1.o %t2.o -o %t
// RUN: llvm-objdump --no-print-imm-hex -d %t | FileCheck %s
// RUN: ld.lld -shared %t1.o %t2.o -o %t.so
// RUN: llvm-objdump --no-print-imm-hex -d %t.so | FileCheck --check-prefix=DSO %s
// RUN: llvm-readelf -S -r %t.so | FileCheck -check-prefix=DSOREL %s

// RUN: llvm-mc -filetype=obj -arm-add-build-attributes -triple=thumbv8-none-linux-gnueabi --arch=thumbeb --mcpu=cortex-m33 %p/Inputs/arm-plt-reloc.s -o %t1.be.o
// RUN: llvm-mc -filetype=obj -arm-add-build-attributes -triple=thumbv8-none-linux-gnueabi --arch=thumbeb --mcpu=cortex-m33 %s -o %t2.be.o
// RUN: ld.lld %t1.be.o %t2.be.o -o %t.be
// RUN: llvm-objdump --no-print-imm-hex -d %t.be | FileCheck %s
// RUN: ld.lld -shared %t1.be.o %t2.be.o -o %t.so.be
// RUN: llvm-objdump --no-print-imm-hex -d %t.so.be | FileCheck --check-prefix=DSO %s
// RUN: llvm-readelf -S -r %t.so.be | FileCheck -check-prefix=DSOREL %s

// RUN: ld.lld --be8 %t1.be.o %t2.be.o -o %t.be
// RUN: llvm-objdump --no-print-imm-hex -d %t.be | FileCheck %s
// RUN: ld.lld --be8 -shared %t1.be.o %t2.be.o -o %t.so.be
// RUN: llvm-objdump --no-print-imm-hex -d %t.so.be | FileCheck --check-prefix=DSO %s
// RUN: llvm-readelf -S -r %t.so.be | FileCheck -check-prefix=DSOREL %s

/// Test PLT entry generation
 .text
 .align 2
 .globl _start
 .type  _start,%function
_start:
 bl func1
 bl func2
 bl func3
 b.w func1
 b.w func2
 b.w func3
 beq.w func1
 beq.w func2
 beq.w func3

/// Executable, expect no PLT
// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <func1>:
// CHECK-NEXT:   bx      lr
// CHECK: <func2>:
// CHECK-NEXT:   bx      lr
// CHECK: <func3>:
// CHECK-NEXT:   bx      lr
// CHECK-NEXT:   d4d4 
// CHECK: <_start>:
// CHECK-NEXT:   bl      {{.*}} <func1>
// CHECK-NEXT:   bl      {{.*}} <func2>
// CHECK-NEXT:   bl      {{.*}} <func3>
// CHECK-NEXT:   b.w     {{.*}} <func1>
// CHECK-NEXT:   b.w     {{.*}} <func2>
// CHECK-NEXT:   b.w     {{.*}} <func3>
// CHECK-NEXT:   beq.w	 {{.*}} <func1>
// CHECK-NEXT:   beq.w	 {{.*}} <func2>
// CHECK-NEXT:   beq.w	 {{.*}} <func3>

// DSO: Disassembly of section .text:
// DSO-EMPTY:
// DSO-NEXT: <func1>:
// DSO-NEXT:     bx      lr
// DSO: <func2>:
// DSO-NEXT:     bx      lr
// DSO: <func3>:
// DSO-NEXT:     bx      lr
// DSO-NEXT:     d4d4 
// DSO: <_start>:
/// 0x10260 = PLT func1
// DSO-NEXT:     bl     0x10260
/// 0x10270 = PLT func2
// DSO-NEXT:     bl     0x10270
/// 0x10280 = PLT func3
// DSO-NEXT:     bl     0x10280
/// 0x10260 = PLT func1
// DSO-NEXT:     b.w    0x10260
/// 0x10270 = PLT func2
// DSO-NEXT:     b.w    0x10270
/// 0x10280 = PLT func3
// DSO-NEXT:     b.w    0x10280
/// 0x10260 = PLT func1
// DSO-NEXT:     beq.w	 0x10260
/// 0x10270 = PLT func2
// DSO-NEXT:     beq.w	 0x10270
/// 0x10280 = PLT func3
// DSO-NEXT:     beq.w	 0x10280
// DSO: Disassembly of section .plt:
// DSO-EMPTY:
// DSO-NEXT: 10240 <.plt>:
// DSO-NEXT:     push    {lr}
// DSO-NEXT:     ldr.w   lr, [pc, #8]
// DSO-NEXT:     add     lr, pc
// DSO-NEXT:     ldr     pc, [lr, #8]!
/// 0x20098 = .got.plt (0x302D8) - pc (0x10238 = .plt + 8) - 8
// DSO-NEXT:     .word   0x00020098
// DSO-NEXT:     .word   0xd4d4d4d4
// DSO-NEXT:     .word   0xd4d4d4d4
// DSO-NEXT:     .word   0xd4d4d4d4
// DSO-NEXT:     .word   0xd4d4d4d4

/// 136 + 2 << 16 + 0x1026c = 0x302f4 = got entry 1
// DSO-NEXT:     10260:       f240 0c88     movw    r12, #136
// DSO-NEXT:                  f2c0 0c02     movt    r12, #2
// DSO-NEXT:                  44fc          add     r12, pc
// DSO-NEXT:                  f8dc f000     ldr.w   pc, [r12]
// DSO-NEXT:                  e7fc          b       0x1026a
/// 124 + 2 << 16 + 0x1027c = 0x302f8 = got entry 2
// DSO-NEXT:     10270:       f240 0c7c     movw    r12, #124
// DSO-NEXT:                  f2c0 0c02     movt    r12, #2
// DSO-NEXT:                  44fc          add     r12, pc
// DSO-NEXT:                  f8dc f000     ldr.w   pc, [r12]
// DSO-NEXT:                  e7fc          b       0x1027a
/// 112 + 2 << 16 + 0x1028c = 0x302fc = got entry 3
// DSO-NEXT:     10280:       f240 0c70     movw    r12, #112
// DSO-NEXT:                  f2c0 0c02     movt    r12, #2
// DSO-NEXT:                  44fc          add     r12, pc
// DSO-NEXT:                  f8dc f000     ldr.w   pc, [r12]
// DSO-NEXT:                  e7fc          b       0x1028a

// DSOREL: .got.plt PROGBITS 000302e8 {{.*}} 000018 00  WA  0   0  4
// DSOREL: Relocation section '.rel.plt'
// DSOREL: 000302f4 {{.*}} R_ARM_JUMP_SLOT {{.*}} func1
// DSOREL: 000302f8 {{.*}} R_ARM_JUMP_SLOT {{.*}} func2
// DSOREL: 000302fc {{.*}} R_ARM_JUMP_SLOT {{.*}} func3
