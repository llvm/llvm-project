// Test PLT section parsing on arm.

// REQUIRES: ld.lld

// RUN: %clang --target=armv6a-none-linux-gnueabi -fuse-ld=lld \
// RUN:   -nostdlib -nostdinc -shared %s -o %t.v6a
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v6a | FileCheck %s --check-prefixes=CHECK,LE

// Test PLT section parsing on armeb.

// RUN: %clang --target=armv6aeb-none-linux-gnueabi -fuse-ld=lld \
// RUN:   -nostdlib -nostdinc -shared %s -o %t.v6aeb
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v6aeb | FileCheck %s --check-prefixes=CHECK,BE
// RUN: obj2yaml %t.v6aeb | FileCheck %s --check-prefixes=NOBE8

// Test PLT section parsing on armeb with be8.

// RUN: %clang --target=armv7aeb-none-linux-gnueabi -fuse-ld=lld \
// RUN:   -nostdlib -nostdinc -shared %s -o %t.v7aeb
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v7aeb | FileCheck %s --check-prefixes=CHECK,BE
// RUN: obj2yaml %t.v7aeb | FileCheck %s --check-prefixes=BE8

// LE: file format elf32-littlearm
// BE: file format elf32-bigarm
// NOBE8-NOT: EF_ARM_BE8
// BE8: EF_ARM_BE8

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT:    <_start>:
// CHECK-NEXT:      push	{r11, lr}
// CHECK-NEXT:      mov	r11, sp
// CHECK-NEXT:      bl	{{.*}} <func1@plt>
// CHECK-NEXT:      bl	{{.*}} <func2@plt>
// CHECK-NEXT:      bl	{{.*}} <func3@plt>

// CHECK: Disassembly of section .plt:
// CHECK:      <func1@plt>:
// CHECK-NEXT:      add	r12, pc, #0, #12
// CHECK-NEXT:      add	r12, r12, #32, #20
// CHECK-NEXT:      ldr	pc, [r12, #132]!
// CHECK-NEXT:      .word	0xd4d4d4d4
// CHECK-EMPTY:
// CHECK-NEXT: <func2@plt>:
// CHECK-NEXT:      add	r12, pc, #0, #12
// CHECK-NEXT:      add	r12, r12, #32, #20
// CHECK-NEXT:      ldr	pc, [r12, #120]!
// CHECK-NEXT:      .word	0xd4d4d4d4
// CHECK-EMPTY:
// CHECK-NEXT: <func3@plt>:
// CHECK-NEXT:      add	r12, pc, #0, #12
// CHECK-NEXT:      add	r12, r12, #32, #20
// CHECK-NEXT:      ldr	pc, [r12, #108]!
// CHECK-NEXT:      .word	0xd4d4d4d4

// Test PLT section parsing on thumb.

// RUN: %clang --target=thumbv8.1m.main-none-linux-eabi \
// RUN:   -c %s -o %t.v8.o
// RUN: ld.lld --shared %t.v8.o -o %t.v8
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v8 | FileCheck %s --check-prefixes=THUMB,LE

// Test PLT section parsing on thumbeb.

// RUN: %clang --target=thumbebv8.1m.main-none-linux-eabi \
// RUN:   -c %s -o %t.v8eb.o
// RUN: ld.lld --shared %t.v8eb.o -o %t.v8eb
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v8eb | FileCheck %s --check-prefixes=THUMB,BE
// RUN: obj2yaml %t.v8eb | FileCheck %s --check-prefixes=NOBE8

// Test PLT section parsing on thumbeb with be8.

// RUN: %clang --target=thumbebv8.1m.main-none-linux-eabi \
// RUN:   -c %s -o %t.v8eb.be8.o
// RUN: ld.lld --shared --be8 %t.v8eb.be8.o -o %t.v8eb.be8
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v8eb.be8 | FileCheck %s --check-prefixes=THUMB,BE
// RUN: obj2yaml %t.v8eb.be8 | FileCheck %s --check-prefixes=BE8

// THUMB: Disassembly of section .text:
// THUMB-EMPTY:
// THUMB-NEXT: <_start>:
// THUMB-NEXT:      push	{r7, lr}
// THUMB-NEXT:      mov r7, sp
// THUMB-NEXT:      bl	{{.*}} <func1@plt>
// THUMB-NEXT:      bl	{{.*}} <func2@plt>
// THUMB-NEXT:      bl	{{.*}} <func3@plt>

// THUMB: Disassembly of section .plt:
// THUMB-EMPTY:
// THUMB:      <func1@plt>:
// THUMB-NEXT:      movw	r12, #136
// THUMB-NEXT:      movt	r12, #2
// THUMB-NEXT:      add	r12, pc
// THUMB-NEXT:      ldr.w	pc, [r12]
// THUMB-NEXT:      b	0x
// THUMB-EMPTY:
// THUMB-NEXT: <func2@plt>:
// THUMB-NEXT:      movw	r12, #124
// THUMB-NEXT:      movt	r12, #2
// THUMB-NEXT:      add	r12, pc
// THUMB-NEXT:      ldr.w	pc, [r12]
// THUMB-NEXT:      b	0x
// THUMB-EMPTY:
// THUMB-NEXT: <func3@plt>:
// THUMB-NEXT:      movw	r12, #112
// THUMB-NEXT:      movt	r12, #2
// THUMB-NEXT:      add	r12, pc
// THUMB-NEXT:      ldr.w	pc, [r12]
// THUMB-NEXT:      b	0x

// Test PLT section with long entries parsing on arm.

// RUN: echo "SECTIONS { \
// RUN:       .text 0x1000 : { *(.text) } \
// RUN:       .plt  0x2000 : { *(.plt) *(.plt.*) } \
// RUN:       .got.plt 0x9000000 : { *(.got.plt) } \
// RUN:       }" > %t.long.script

// RUN: %clang --target=armv6a-none-linux-gnueabi -fuse-ld=lld \
// RUN:   -Xlinker --script=%t.long.script -nostdlib -nostdinc \
// RUN:   -shared %s -o %t.v6a.long
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v6a.long | FileCheck %s --check-prefixes=CHECKLONG,LE

// Test PLT section with long entries parsing on armeb.

// RUN: %clang --target=armv6aeb-none-linux-gnueabi -fuse-ld=lld \
// RUN:   -Xlinker --script=%t.long.script -nostdlib -nostdinc \
// RUN:   -shared %s -o %t.v6aeb.long
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v6aeb.long | FileCheck %s --check-prefixes=CHECKLONG,BE
// RUN: obj2yaml %t.v6aeb.long | FileCheck %s --check-prefixes=NOBE8

// Test PLT section with long entries parsing on armeb with be8.

// RUN: %clang --target=armv7aeb-none-linux-gnueabi -fuse-ld=lld \
// RUN:   -Xlinker --script=%t.long.script -nostdlib -nostdinc \
// RUN:   -shared %s -o %t.v7aeb.long
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v7aeb.long | FileCheck %s --check-prefixes=CHECKLONG,BE
// RUN: obj2yaml %t.v7aeb.long | FileCheck %s --check-prefixes=BE8

// CHECKLONG:       Disassembly of section .text:
// CHECKLONG-EMPTY:
// CHECKLONG-NEXT:  <_start>:
// CHECKLONG-NEXT:      push	{r11, lr}
// CHECKLONG-NEXT:      mov	r11, sp
// CHECKLONG-NEXT:      bl	0x2020 <func1@plt>
// CHECKLONG-NEXT:      bl	0x2030 <func2@plt>
// CHECKLONG-NEXT:      bl	0x2040 <func3@plt>

// CHECKLONG:       Disassembly of section .plt:
// CHECKLONG:       00002020 <func1@plt>:
// CHECKLONG-NEXT:      ldr	r12, [pc, #4]
// CHECKLONG-NEXT:      add	r12, r12, pc
// CHECKLONG-NEXT:      ldr	pc, [r12]
// CHECKLONG-NEXT:      .word	0x08ffdfe0
// CHECKLONG-EMPTY:
// CHECKLONG-NEXT:  00002030 <func2@plt>:
// CHECKLONG-NEXT:      ldr	r12, [pc, #4]
// CHECKLONG-NEXT:      add	r12, r12, pc
// CHECKLONG-NEXT:      ldr	pc, [r12]
// CHECKLONG-NEXT:      .word	0x08ffdfd4
// CHECKLONG-EMPTY:
// CHECKLONG-NEXT:  00002040 <func3@plt>:
// CHECKLONG-NEXT:      ldr	r12, [pc, #4]
// CHECKLONG-NEXT:      add	r12, r12, pc
// CHECKLONG-NEXT:      ldr	pc, [r12]
// CHECKLONG-NEXT:      .word	0x08ffdfc8

// Test PLT section with mixed long and short entries parsing on arm.

// RUN: echo "SECTIONS { \
// RUN:       .text 0x1000 : { *(.text) } \
// RUN:       .plt  0x2000 : { *(.plt) *(.plt.*) } \
// RUN:       .got.plt 0x8002020 : { *(.got.plt) } \
// RUN:       }" > %t.mix.script

// RUN: %clang --target=armv6a-none-linux-gnueabi -fuse-ld=lld \
// RUN:   -Xlinker --script=%t.mix.script -nostdlib -nostdinc \
// RUN:   -shared %s -o %t.v6a.mix
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v6a.mix | FileCheck %s --check-prefixes=CHECKMIX,LE

// Test PLT section with mixed long and short entries parsing on armeb.

// RUN: %clang --target=armv6aeb-none-linux-gnueabi -fuse-ld=lld \
// RUN:   -Xlinker --script=%t.mix.script -nostdlib -nostdinc \
// RUN:   -shared %s -o %t.v6aeb.mix
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v6aeb.mix | FileCheck %s --check-prefixes=CHECKMIX,BE
// RUN: obj2yaml %t.v6aeb.mix | FileCheck %s --check-prefixes=NOBE8

// Test PLT section with mixed long and short entries parsing on armeb with be8.

// RUN: %clang --target=armv7aeb-none-linux-gnueabi -fuse-ld=lld \
// RUN:   -Xlinker --script=%t.mix.script -nostdlib -nostdinc \
// RUN:   -shared %s -o %t.v7aeb.mix
// RUN: llvm-objdump --no-show-raw-insn --no-print-imm-hex \
// RUN:   -d %t.v7aeb.mix | FileCheck %s --check-prefixes=CHECKMIX,BE
// RUN: obj2yaml %t.v7aeb.mix | FileCheck %s --check-prefixes=BE8

// CHECKMIX:        Disassembly of section .text:
// CHECKMIX-EMPTY:
// CHECKMIX-NEXT:   <_start>:
// CHECKMIX-NEXT:       push	{r11, lr}
// CHECKMIX-NEXT:       mov	r11, sp
// CHECKMIX-NEXT:       bl	0x2020 <func1@plt>
// CHECKMIX-NEXT:       bl	0x2030 <func2@plt>
// CHECKMIX-NEXT:       bl	0x2040 <func3@plt>

// CHECKMIX:        Disassembly of section .plt:
// CHECKMIX:        00002020 <func1@plt>:
// CHECKMIX-NEXT:       ldr	r12, [pc, #4]
// CHECKMIX-NEXT:       add	r12, r12, pc
// CHECKMIX-NEXT:       ldr	pc, [r12]
// CHECKMIX-NEXT:       .word	0x08000000
// CHECKMIX-EMPTY:
// CHECKMIX-NEXT:   00002030 <func2@plt>:
// CHECKMIX-NEXT:       add	r12, pc, #133169152
// CHECKMIX-NEXT:       add	r12, r12, #1044480
// CHECKMIX-NEXT:       ldr	pc, [r12, #4088]!
// CHECKMIX-NEXT:       .word	0xd4d4d4d4
// CHECKMIX-EMPTY:
// CHECKMIX-NEXT:   00002040 <func3@plt>:
// CHECKMIX-NEXT:       add	r12, pc, #133169152
// CHECKMIX-NEXT:       add	r12, r12, #1044480
// CHECKMIX-NEXT:       ldr	pc, [r12, #4076]!
// CHECKMIX-NEXT:       .word	0xd4d4d4d4

extern void *func1();
extern void *func2();
extern void *func3();

void _start() {
  func1();
  func2();
  func3();
}
