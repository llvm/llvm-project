# REQUIRES: mips
# Check PLT creation for microMIPS to microMIPS calls.

# RUN: echo "SECTIONS { \
# RUN:         . = 0x20000;  .text ALIGN(0x100) : { *(.text) } \
# RUN:         . = 0x20300;  .plt : { *(.plt) } \
# RUN:       }" > %t.script

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %S/Inputs/mips-micro.s -o %t1eb.o
# RUN: ld.lld -shared -soname=teb.so -o %teb.so %t1eb.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %s -o %t2eb.o
# RUN: ld.lld --script %t.script -o %teb.exe %t2eb.o %teb.so
# RUN: llvm-objdump -d --mattr=micromips --no-show-raw-insn %teb.exe \
# RUN:   | FileCheck --check-prefix=R2 %s
# RUN: llvm-readelf -A %teb.exe | FileCheck --check-prefix=PLT %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -mattr=micromips %S/Inputs/mips-micro.s -o %t1el.o
# RUN: ld.lld -shared -soname=tel.so -o %tel.so %t1el.o
# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -mattr=micromips %s -o %t2el.o
# RUN: ld.lld --script %t.script -o %tel.exe %t2el.o %tel.so
# RUN: llvm-objdump -d --mattr=micromips --no-show-raw-insn %tel.exe \
# RUN:   | FileCheck --check-prefix=R2 %s
# RUN: llvm-readelf -A %tel.exe | FileCheck --check-prefix=PLT %s

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips -mcpu=mips32r6 %S/Inputs/mips-micro.s -o %t1eb.o
# RUN: ld.lld -shared -soname=teb.so -o %teb.so %t1eb.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips -mcpu=mips32r6 %s -o %t2eb.o
# RUN: ld.lld --script %t.script -o %teb.exe %t2eb.o %teb.so
# RUN: llvm-objdump -d --mattr=micromips %teb.exe --no-show-raw-insn \
# RUN: | FileCheck --check-prefix=R6 %s

# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -mattr=micromips -mcpu=mips32r6 %S/Inputs/mips-micro.s -o %t1el.o
# RUN: ld.lld -shared -soname=tel.so -o %tel.so %t1el.o
# RUN: llvm-mc -filetype=obj -triple=mipsel-unknown-linux \
# RUN:         -mattr=micromips -mcpu=mips32r6 %s -o %t2el.o
# RUN: ld.lld --script %t.script -o %tel.exe %t2el.o %tel.so
# RUN: llvm-objdump -d --mattr=micromips --no-show-raw-insn %tel.exe \
# RUN:   | FileCheck --check-prefix=R6 %s

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %S/Inputs/mips-micro.s -o %t1eb.o
# RUN: ld.lld -shared -soname=teb.so -o %teb.so %t1eb.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %S/Inputs/mips-fpic.s -o %t-reg.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %s -o %t2eb.o
# RUN: ld.lld --script %t.script -o %teb.exe %t-reg.o %t2eb.o %teb.so
# RUN: llvm-objdump -d --mattr=micromips --no-show-raw-insn %teb.exe \
# RUN:   | FileCheck --check-prefix=R2 %s

# R2:      .plt:
# R2:         20300:  addiupc $3, 52
# R2-NEXT:            lw      $25, 0($3)
# R2-NEXT:            subu16  $2, $2, $3
# R2-NEXT:            srl16   $2, $2, 2
# R2-NEXT:            addiu   $24, $2, -2
# R2-NEXT:            move    $15, $ra
# R2-NEXT:            jalrs16 $25
# R2-NEXT:            move    $gp, $3
# R2-NEXT:            nop
# R2-NEXT:            ...
# R2-NEXT:    20320:  addiupc $2, 28
# R2-NEXT:            lw      $25, 0($2)
# R2-NEXT:            jr16    $25
# R2-NEXT:            move    $24, $2

# R6:      .plt:
# R6:         20300:  lapc    $3, 52
# R6-NEXT:            lw      $25, 0($3)
# R6-NEXT:            subu16  $2, $2, $3
# R6-NEXT:            srl16   $2, $2, 2
# R6-NEXT:            addiu   $24, $2, -2
# R6-NEXT:            move16  $15, $ra
# R6-NEXT:            move16  $gp, $3
# R6-NEXT:            jalr    $25

# R6:         20320:  lapc    $2, 28
# R6-NEXT:            lw      $25, 0($2)
# R6-NEXT:            move16  $24, $2
# R6-NEXT:            jrc16   $25

# PLT:      PLT GOT:
# PLT:       Entries:
# PLT-NEXT:    Address  Initial Sym.Val. Type    Ndx Name
# PLT-NEXT:   0002033c 00020301 00000000 FUNC    UND foo
#             ^ 0x20320 + 28

  .text
  .set micromips
  .global __start
__start:
  jal foo
