# REQUIRES: system-linux

# Check that references and the symbol table entry for __EH_FRAME_BEGIN__ are
# updated to the regenerated .eh_frame section.

# RUN: llvm-mc -filetype=obj -triple riscv64-unknown-linux %s -o %t.o
# RUN: ld.lld -q %t.o -o %t.exe
# RUN: llvm-readelf -SW -s %t.exe | FileCheck %s --check-prefix=INPUT
# RUN: llvm-bolt %t.exe -o %t.bolt.exe
# RUN: llvm-readelf -SW -s %t.bolt.exe > %t.out
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt.exe >> %t.out
# RUN: FileCheck %s < %t.out

  .section .eh_frame,"a",@progbits
  # Keep a distinct CIE/FDE before the registration anchor so that the input
  # symbol has a nonzero section offset.
  .long 0x10
  .long 0
  .byte 0x01, 0x7a, 0x52, 0x00
  # Use a non-default code alignment to prevent CIE merging.
  .byte 0x02, 0x78, 0x01, 0x01
  .byte 0x1b, 0x0c, 0x02, 0x00
  .long 0x14
  .long 0x18
.Ldummy_fde_pc:
  .long 0
  .reloc .Ldummy_fde_pc, R_RISCV_32_PCREL, dummy
  .dword 4
  .long 0

  # This symbol is local in GNU crtbegin and is uniquified internally by BOLT.
  .type __EH_FRAME_BEGIN__,@object
__EH_FRAME_BEGIN__:

  .text
  .type dummy,@function
dummy:
  ret
  .size dummy, .-dummy

  .globl _start
  .type _start,@function
_start:
  .cfi_startproc
.Lpcrel_hi:
  auipc a0, %pcrel_hi(__EH_FRAME_BEGIN__)
  addi a0, a0, %pcrel_lo(.Lpcrel_hi)
  ret
  .reloc 0, R_RISCV_NONE
  .cfi_endproc
  .size _start, .-_start

# INPUT: ] .eh_frame PROGBITS [[#%x, INPUT_EH_FRAME_ADDR:]]
# INPUT: [[#INPUT_EH_FRAME_ADDR + 0x2c]] 0 OBJECT LOCAL DEFAULT {{[0-9]+}} __EH_FRAME_BEGIN__

# CHECK: ] .eh_frame PROGBITS [[#%x, EH_FRAME_ADDR:]]
# CHECK: [[#EH_FRAME_ADDR]] 0 OBJECT LOCAL DEFAULT {{[0-9]+}} __EH_FRAME_BEGIN__
# CHECK: [[#%x, START_ADDR:]] <_start>:
# CHECK-NEXT: [[#START_ADDR]]: auipc a0, 0x0
# CHECK-NEXT: [[#START_ADDR + 4]]: addi a0, a0, 0x[[#EH_FRAME_ADDR - START_ADDR]]
