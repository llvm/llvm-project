# REQUIRES: mips
# Check R_MIPS_GOT16 relocation calculation.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x20000; .text : { *(.text) } \
# RUN:         . = 0x30000; .got  : { *(.got)  } \
# RUN:       }" > %t.script

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t-be.o
# RUN: ld.lld %t-be.o --script %t.script -o %t-be.exe
# RUN: llvm-readobj --sections -r --symbols -A %t-be.exe \
# RUN:   | FileCheck -check-prefix=ELF %s
# RUN: llvm-objdump -d %t-be.exe | FileCheck --check-prefix=DIS %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux %s -o %t-el.o
# RUN: ld.lld %t-el.o --script %t.script -o %t-el.exe
# RUN: llvm-readobj --sections -r --symbols -A %t-el.exe \
# RUN:   | FileCheck -check-prefix=ELF %s
# RUN: llvm-objdump -d %t-el.exe | FileCheck --check-prefix=DIS %s

# RUN: ld.lld -shared %t-be.o --script %t.script -o %t-be.so
# RUN: llvm-readobj --sections -r --symbols -A %t-be.so \
# RUN:   | FileCheck -check-prefix=ELF %s
# RUN: llvm-objdump -d %t-be.so | FileCheck --check-prefix=DIS %s

# RUN: ld.lld -shared %t-el.o --script %t.script -o %t-el.so
# RUN: llvm-readobj --sections -r --symbols -A %t-el.so \
# RUN:   | FileCheck -check-prefix=ELF %s
# RUN: llvm-objdump -d %t-el.so | FileCheck --check-prefix=DIS %s

  .text
  .globl  __start
__start:
  lui $2, %got(v1)

  .data
  .globl v1
v1:
  .word 0

# ELF:      Section {
# ELF:        Name: .got
# ELF:        Flags [
# ELF-NEXT:     SHF_ALLOC
# ELF-NEXT:     SHF_MIPS_GPREL
# ELF-NEXT:     SHF_WRITE
# ELF-NEXT:   ]
#
# ELF:      Relocations [
# ELF-NEXT: ]
#
# ELF:      Symbol {
# ELF:        Name: v1
# ELF-NEXT:   Value: 0x[[V1:[0-9A-F]+]]
#
# ELF:      {{.*}} GOT {
# ELF-NEXT:   Canonical gp value: 0x37FF0
#
# ELF:        Entry {
# ELF:          Address: 0x30008
# ELF-NEXT:     Access: -32744
# ELF-NEXT:     Initial: 0x[[V1]]

# "v1 GOT entry address" - _gp
# 0x30008 - 0x37FF0 = -0x7fe8 == 0x8018 == 32792
# DIS:  lui $2, 32792
