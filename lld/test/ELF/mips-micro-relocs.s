# REQUIRES: mips
# Check handling of microMIPS relocations.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x20000;  .text ALIGN(0x100) : { *(.text) } \
# RUN:         . = 0x20300;  .plt  : { *(.plt) } \
# RUN:         . = 0x30000;  .data : { *(.data) } \
# RUN:       }" > %t.script

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %S/Inputs/mips-micro.s -o %t1eb.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %s -o %t2eb.o
# RUN: ld.lld -o %teb.exe -script %t.script %t1eb.o %t2eb.o
# RUN: llvm-objdump -d -t -s -mattr=micromips --no-show-raw-insn %teb.exe \
# RUN:   | FileCheck --check-prefixes=ASM,EB %s
# RUN: llvm-readelf -h %teb.exe | FileCheck --check-prefix=ELF %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -mattr=micromips %S/Inputs/mips-micro.s -o %t1el.o
# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -mattr=micromips %s -o %t2el.o
# RUN: ld.lld -o %tel.exe -script %t.script %t1el.o %t2el.o
# RUN: llvm-objdump -d -t -s -mattr=micromips --no-show-raw-insn %tel.exe \
# RUN:   | FileCheck --check-prefixes=ASM,EL %s
# RUN: llvm-readelf -h %tel.exe | FileCheck --check-prefix=ELF %s

# ASM: 00038000         .got   00000000 .hidden _gp
# ASM: 00020100 g F     .text  00000000 0x80 foo
# ASM: 00020110         .text  00000000 0x80 __start

# EB:      Contents of section .data:
# EB-NEXT:  30000 fffe8111

# EB:      Contents of section .debug_info
# EB-NEXT:  0000 00020111

# EL:      Contents of section .data:
# EL-NEXT:  30000 1181feff

# EL:      Contents of section .debug_info
# EL-NEXT:  0000 11010200

# ASM:      __start:
# ASM-NEXT:      20110:  lui     $3, 1
# ASM-NEXT:              addiu   $3, $3, 32495
# ASM-NEXT:              lw      $3, -32744($gp)
# ASM-NEXT:              lw      $3, -32744($3)
# ASM-NEXT:              beqz16  $6, -32
# ASM-NEXT:              sll     $3, $fp, 0
# ASM-NEXT:              b16     -40
# ASM-NEXT:              nop
# ASM-NEXT:              b       -44

# ELF: Entry point address: 0x20111

  .text
  .set micromips
  .global __start
__start:
  lui     $3, %hi(_gp_disp)       # R_MICROMIPS_HI16
  addiu   $3, $3, %lo(_gp_disp)   # R_MICROMIPS_LO16

  lw      $3, %call16(foo)($gp)   # R_MICROMIPS_CALL16
  lw      $3, %got(foo)($3)       # R_MICROMIPS_GOT16

  beqz16  $6, foo                 # R_MICROMIPS_PC7_S1
  b16     foo                     # R_MICROMIPS_PC10_S1
  b       foo                     # R_MICROMIPS_PC16_S1

  .data
  .gpword __start                 # R_MIPS_GPREL32

  .section .debug_info
  .word __start                   # R_MIPS_32
