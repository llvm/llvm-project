/// Test that R_ARM_SBREL32 relocations in debug info are relocated as if the
/// static base register (r9) is zero. Real DWARF info will use an expression to
/// add this to the real value of the static base at runtime.

// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=armv7a asm.s -o obj.o
// RUN: ld.lld -T lds.ld obj.o -o exe.elf 2>&1 | FileCheck %s --implicit-check-not=warning: --allow-empty
// RUN: llvm-objdump -D exe.elf | FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <rw>:
// DISASM-NEXT:      1000: 0000002a

// DISASM-LABEL: <rw2>:
// DISASM-NEXT:      2000: 000004d2

// DISASM-LABEL: <.debug_something>:
// DISASM-NEXT:         0: 00001000
// DISASM-NEXT:        ...
// DISASM-NEXT:       104: 00002000

//--- lds.ld
SECTIONS {
  data1 0x1000 : { *(data1) }
  data2 0x2000 : { *(data2) }
}

//--- asm.s
  .text
  .type _start,%function
  .globl  _start
_start:
  bx lr
  .size _start, .-_start

  .section data1, "aw", %progbits
  .type rw,%object
  .globl  rw
rw:
  .long 42
  .size rw, 4

  .section data2, "aw", %progbits
  .type rw2,%object
  .globl  rw2
rw2:
  .long 1234
  .size rw2, 4

  .section  .debug_something, "", %progbits
  .long rw(sbrel)
  .space 0x100
  .long rw2(sbrel)
