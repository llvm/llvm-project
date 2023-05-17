// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %p/Inputs/relocation-copy-arm.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname fixed-length-string.so -o %t2.so
// RUN: ld.lld --hash-style=sysv %t.o %t2.so -o %t3
// RUN: llvm-readobj -S -r --expand-relocs --symbols %t3 | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d --triple=armv7a-none-linux-gnueabi --no-show-raw-insn %t3 | FileCheck --check-prefix=CODE %s
// RUN: llvm-objdump -s --triple=armv7a-none-linux-gnueabi --section=.rodata %t3 | FileCheck --check-prefix=RODATA %s

// Copy relocations R_ARM_COPY are required for y and z
 .syntax unified
 .text
 .globl _start
_start:
 movw r2,:lower16: y
 movt r2,:upper16: y
 ldr r3,[pc,#4]
 ldr r3,[r3,#0]
 .rodata
 .word z

// CHECK:     Name: .bss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x40220
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 8
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 16

// CHECK: Relocations [
// CHECK-NEXT:  Section {{.*}} .rel.dyn {
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x40220
// CHECK-NEXT:      Type: R_ARM_COPY
// CHECK-NEXT:      Symbol: y
// CHECK-NEXT:    }
// CHECK-NEXT:    Relocation {
// CHECK-NEXT:      Offset: 0x40224
// CHECK-NEXT:      Type: R_ARM_COPY
// CHECK-NEXT:      Symbol: z
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK: Symbols [
// CHECK:     Name: y
// CHECK-NEXT:    Value: 0x40220
// CHECK-NEXT:    Size: 4
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: Object
// CHECK-NEXT:    Other:
// CHECK-NEXT:    Section: .bss
// CHECK:    Name: z
// CHECK-NEXT:    Value: 0x40224
// CHECK-NEXT:    Size: 4
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: Object
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .bss

// CODE: Disassembly of section .text:
// CODE-EMPTY:
// CODE-NEXT: <_start>:
/// S + A = 0x40220 + 0 = 65536 * 1 + 544
// CODE-NEXT:   201b4:       movw    r2, #544
// CODE-NEXT:   201b8:       movt    r2, #4
// CODE-NEXT:   201bc:       ldr     r3, [pc, #4]
// CODE-NEXT:   201c0:       ldr     r3, [r3]


// RODATA: Contents of section .rodata:
/// S(z) = 0x40004
// RODATA-NEXT: 101b0 24020400
