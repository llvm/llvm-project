// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=armv7-none-eabi code.s -o code.o
// RUN: ld.lld -T unsigned1.ld code.o -o unsigned1.elf
// RUN: llvm-objdump --triple=armv7 --no-show-raw-insn -d unsigned1.elf | FileCheck %s --check-prefix=UNSIGNED1
// RUN: ld.lld -T unsigned2.ld code.o -o unsigned2.elf
// RUN: llvm-objdump --triple=armv7 --no-show-raw-insn -d unsigned2.elf | FileCheck %s --check-prefix=UNSIGNED2
// RUN: ld.lld -T signed1.ld code.o -o signed1.elf
// RUN: llvm-objdump --triple=armv7 --no-show-raw-insn -d signed1.elf | FileCheck %s --check-prefix=SIGNED1
// RUN: ld.lld -T signed2.ld code.o -o signed2.elf
// RUN: llvm-objdump --triple=armv7 --no-show-raw-insn -d signed2.elf | FileCheck %s --check-prefix=SIGNED2

/// The aim of this test is to ensure that a BL instruction near one end of the
/// address space can reach a function at the extreme other end, directly,
/// using a branch offset that makes the address wrap round. We check this at
/// both the unsigned wraparound point (one address near 0 and the other near
/// 0xFFFFFFFF) and the signed wraparound point (addresses either side of
/// 0x80000000), crossing the boundary in both directions. In all four cases we
/// expect a direct branch with no veneer.

// UNSIGNED1: Disassembly of section .text.lowaddr:
// UNSIGNED1: <func>:
// UNSIGNED1:    10000:       bx      lr
//
// UNSIGNED1: Disassembly of section .text.highaddr:
// UNSIGNED1: <_start>:
// UNSIGNED1: ffff0000:       bl      0x10000
// UNSIGNED1-NEXT:            bx      lr

// UNSIGNED2: Disassembly of section .text.lowaddr:
// UNSIGNED2: <_start>:
// UNSIGNED2:    10000:       bl      0xffff0000
// UNSIGNED2-NEXT:            bx      lr
//
// UNSIGNED2: Disassembly of section .text.highaddr:
// UNSIGNED2: <func>:
// UNSIGNED2: ffff0000:       bx      lr

// SIGNED1:   Disassembly of section .text.posaddr:
// SIGNED1:   <_start>:
// SIGNED1:   7fff0000:       bl      0x80010000
// SIGNED1-NEXT:              bx      lr
//
// SIGNED1:   Disassembly of section .text.negaddr:
// SIGNED1:   <func>:
// SIGNED1:   80010000:       bx      lr

// SIGNED2:   Disassembly of section .text.posaddr:
// SIGNED2:   <func>:
// SIGNED2:   7fff0000:       bx      lr
//
// SIGNED2:   Disassembly of section .text.negaddr:
// SIGNED2:   <_start>:
// SIGNED2:   80010000:       bl      0x7fff0000
// SIGNED2-NEXT:              bx      lr

//--- code.s

  .section .text.callee, "ax", %progbits
  .global func
  .type func, %function
func:
  bx lr

  .section .text.caller, "ax", %progbits
  .global _start
  .type _start, %function
_start:
  bl func
  bx lr

//--- unsigned1.ld

ENTRY(_start)
SECTIONS {
  .text.lowaddr  0x00010000 : AT(0x00010000) { *(.text.callee) }
  .text.highaddr 0xffff0000 : AT(0xffff0000) { *(.text.caller) }
}

//--- unsigned2.ld

ENTRY(_start)
SECTIONS {
  .text.lowaddr  0x00010000 : AT(0x00010000) { *(.text.caller) }
  .text.highaddr 0xffff0000 : AT(0xffff0000) { *(.text.callee) }
}

//--- signed1.ld

ENTRY(_start)
SECTIONS {
  .text.posaddr  0x7fff0000 : AT(0x7fff0000) { *(.text.caller) }
  .text.negaddr  0x80010000 : AT(0x80010000) { *(.text.callee) }
}

//--- signed2.ld

ENTRY(_start)
SECTIONS {
  .text.posaddr  0x7fff0000 : AT(0x7fff0000) { *(.text.callee) }
  .text.negaddr  0x80010000 : AT(0x80010000) { *(.text.caller) }
}
