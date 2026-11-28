## Check that BOLT rewrites DW_AT_entry_pc to the relocated entry instruction.

# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --relocs --update-debug-sections

# RUN: llvm-objdump -d %t.bolt > %t.dump
# RUN: llvm-dwarfdump %t.bolt >> %t.dump
# RUN: FileCheck %s < %t.dump

# CHECK: Disassembly of section .text:
# CHECK-LABEL: <_start>:
# CHECK: [[#%x,ENTRY:]]: {{.*}} b {{.*}} <_start>

# CHECK: DW_TAG_inlined_subroutine
# CHECK-NEXT: DW_AT_abstract_origin {{.*}}"inner")
# CHECK-NEXT: DW_AT_entry_pc (0x{{0*}}[[#ENTRY]])

    .file 1 "debug-entry-pc-fixes.s"
    .type _start,@function
    .globl _start
_start:
  mov w0, #0
.Lentry:
  b _start

## DWARF for one inlined instance of inner with entry PC .Lentry.
  .set .Lcode_size, . - _start

  .section .debug_abbrev,"",@progbits
  .byte 1, 0x11, 1                   // Compile unit, children
  .byte 0x10, 0x17                   // stmt_list (sec_offset)
  .byte 0x11, 0x01, 0x12, 0x06, 0, 0 // low_pc (addr), high_pc (data4), end

  .byte 2, 0x2e, 1                   // Concrete subprogram, children
  .byte 0x03, 0x08                   // name (string)
  .byte 0x11, 0x01, 0x12, 0x06, 0, 0 // low_pc (addr), high_pc (data4), end

  .byte 3, 0x1d, 0                   // Inlined subroutine
  .byte 0x31, 0x13                   // abstract_origin (ref4)
  .byte 0x52, 0x01                   // entry_pc (addr)
  .byte 0x11, 0x01, 0x12, 0x06, 0, 0 // low_pc (addr), high_pc (data4), end

  .byte 4, 0x2e, 0                   // Abstract subprogram
  .byte 0x03, 0x08, 0x20, 0x0b, 0, 0 // name (string), inline (data1), end
  .byte 0                            // End of table

  .section .debug_info,"",@progbits
.Lcu:
  .long .Lcu_end - .Lcu - 4 // Unit length excludes this field.
  .short 5
  .byte 1, 8        // DW_UT_compile, address size
  .long 0           // Abbrev offset

  .byte 1           // Compile unit
  .long 0           // Line table offset
  .quad _start
  .long .Lcode_size

  .byte 2           // Concrete subprogram
  .asciz "_start"
  .quad _start
  .long .Lcode_size

  .byte 3           // Inlined subroutine
  .long .Linner - .Lcu
  .quad .Lentry     // DW_AT_entry_pc
  .quad _start
  .long .Lcode_size
  .byte 0           // End of subprogram children
.Linner:
  .byte 4           // Abstract subprogram
  .asciz "inner"
  .byte 1           // DW_INL_inlined
  .byte 0           // End of compile unit children
.Lcu_end:
