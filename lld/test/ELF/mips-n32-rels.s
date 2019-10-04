# REQUIRES: mips
# Check handling of N32 ABI relocation records.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         -target-abi n32 -o %t.o %s
# RUN: echo "SECTIONS { \
# RUN:         . = 0x20000;  .text ALIGN(0x100) : { *(.text) } \
# RUN:       }" > %t.script
# RUN: ld.lld %t.o -script %t.script -o %t.exe
# RUN: llvm-objdump -t -d -s --no-show-raw-insn %t.exe | FileCheck %s
# RUN: llvm-readelf -h %t.exe | FileCheck -check-prefix=ELF %s

  .option pic2
  .text
  .type   __start, @function
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
  .gpword(loc)                                # R_MIPS_GPREL32

# CHECK: 00020104      .text   00000000 loc
# CHECK: 00028100      .got    00000000 .hidden _gp
# CHECK: 00020100 g  F .text   00000000 __start

# CHECK:      Contents of section .rodata:
# CHECK-NEXT:  {{[0-9a-f]+}} ffff8004
#                            ^-- loc - _gp

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: __start:
# CHECK-NEXT:    20100:  lui     $gp, 1
#                                     ^-- 0x20100 - 0x28100
#                                     ^-- 0 - 0xffff8000
#                                     ^-- %hi(0x8000)
# CHECK:      loc:
# CHECK-NEXT:    20104:  daddiu  $gp, $gp, -32768
#                                          ^-- 0x20100 - 0x28100
#                                          ^-- 0 - 0xffff8000
#                                          ^-- %lo(0x8000)

# ELF: Class:                ELF32
# ELF: Data:                 2's complement, big endian
# ELF: Version:              1 (current)
# ELF: OS/ABI:               UNIX - System V
# ELF: ABI Version:          0x0
# ELF: Type:                 EXEC (Executable file)
# ELF: Machine:              MIPS R3000
# ELF: Version:              0x1
# ELF: Entry point address:  0x20100
# ELF: Flags:                0x60000026, pic, cpic, abi2, mips64
