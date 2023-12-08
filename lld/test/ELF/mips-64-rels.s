# REQUIRES: mips
# Check handling multiple MIPS N64 ABI relocations packed
# into the single relocation record.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:         . = 0x20000; .text ALIGN(0x1000) : { *(.text) } \
# RUN:         . = 0x30000; .got : { *(.got)  } \
# RUN:       }" > %t.script
# RUN: ld.lld %t.o --script %t.script -o %t.exe
# RUN: llvm-objdump -d -s -t --print-imm-hex --no-show-raw-insn %t.exe | FileCheck %s
# RUN: llvm-readobj -r %t.exe | FileCheck -check-prefix=REL %s

# CHECK: 0000000000037ff0 l .got  0000000000000000 .hidden _gp

# CHECK:      Contents of section .rodata:
# CHECK-NEXT:  {{[0-9a-f]+}} ffffffff fffe9014
#                            ^-- 0x21004 - 0x37ff0 = 0xfffffffffffe9014

# CHECK:      <__start>:
# CHECK-NEXT:    21000:  lui     $gp, 0x1
#                                     ^-- 0x21000 - 0x37ff0
#                                     ^-- 0 - 0xffffffffffff9010
#                                     ^-- %hi(0x16ff0)
# CHECK:      <loc>:
# CHECK-NEXT:    21004:  daddiu  $gp, $gp, 0x6ff0
#                                     ^-- 0x21000 - 0x37ff0
#                                     ^-- 0 - 0xfffffffffffe9010
#                                     ^-- %lo(0x16ff0)

# REL:      Relocations [
# REL-NEXT: ]

  .text
  .global  __start
__start:
  lui     $gp,%hi(%neg(%gp_rel(__start)))     # R_MIPS_GPREL16
                                              # R_MIPS_SUB
                                              # R_MIPS_HI16
loc:
  daddiu  $gp,$gp,%lo(%neg(%gp_rel(__start))) # R_MIPS_GPREL16
                                              # R_MIPS_SUB
                                              # R_MIPS_LO16

  .section  .rodata,"a",@progbits
  .gpdword(loc)                               # R_MIPS_GPREL32
                                              # R_MIPS_64
