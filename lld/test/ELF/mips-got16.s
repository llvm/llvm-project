# REQUIRES: mips
# Check R_MIPS_GOT16 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { \
# RUN:         . = 0x1000;  .text ALIGN(0x1000) : { *(.text) } \
# RUN:         . = 0x3000;  .data : { *(.data) } \
# RUN:       }" > %t.script
# RUN: ld.lld %t.o -shared --script %t.script -o %t.so
# RUN: llvm-objdump -d -t --no-show-raw-insn %t.so | FileCheck %s
# RUN: llvm-readelf -r -A %t.so | FileCheck -check-prefix=GOT %s

# CHECK: SYMBOL TABLE:
# CHECK: 00024008 l       .data           00000000 .hidden bar
# CHECK: 00000000         *UND*           00000000 foo

# CHECK:       <__start>:
# CHECK-NEXT:    lw      $8, -32744($gp)
# CHECK-NEXT:    addi    $8, $8, 8236
# CHECK-NEXT:    lw      $8, -32732($gp)
# CHECK-NEXT:    addi    $8, $8, -16384
# CHECK-NEXT:    lw      $8, -32728($gp)
# CHECK-NEXT:    addi    $8, $8, -16380
# CHECK-NEXT:    lw      $8, -32728($gp)
# CHECK-NEXT:    addi    $8, $8, 16388
# CHECK-NEXT:    lw      $8, -32720($gp)
# CHECK-NEXT:    addi    $8, $8, 16392
# CHECK-NEXT:    lw      $8, -32716($gp)

# GOT: There are no relocations in this file.

# GOT:       Local entries:
# GOT-NEXT:        Access  Initial
# GOT-NEXT:   -32744(gp) 00000000
#                        ^-- (0x2000 + 0x8000) & ~0xffff
# GOT-NEXT:   -32740(gp) 00010000
#                        ^-- redundant unused entry
# GOT-NEXT:   -32736(gp) 00000000
#                        ^-- redundant unused entry
# GOT-NEXT:   -32732(gp) 00010000
#                        ^-- (0x3000 + 0x9000 + 0x8000) & ~0xffff
# GOT-NEXT:   -32728(gp) 00020000
#                        ^-- (0x3000 + 0x9000 + 0x10004 + 0x8000) & ~0xffff
#                        ^-- (0x3000 + 0x9000 + 0x18004 + 0x8000) & ~0xffff
# GOT-NEXT:   -32724(gp) 00030000
#                        ^-- redundant unused entry
# GOT-NEXT:   -32720(gp) 00024008
#                        ^-- 'bar' address
# GOT-EMPTY:
# GOT-NEXT:  Global entries:
# GOT-NEXT:       Access  Initial Sym.Val. Type    Ndx Name
# GOT-NEXT:   -32716(gp) 00000000 00000000 NOTYPE  UND foo

  .text
  .globl  __start
__start:
  lw      $t0,%got($LC0)($gp)
  addi    $t0,$t0,%lo($LC0)
  lw      $t0,%got($LC1)($gp)
  addi    $t0,$t0,%lo($LC1)
  lw      $t0,%got($LC1+0x10004)($gp)
  addi    $t0,$t0,%lo($LC1+0x10004)
  lw      $t0,%got($LC1+0x18004)($gp)
  addi    $t0,$t0,%lo($LC1+0x18004)
  lw      $t0,%got(bar)($gp)
  addi    $t0,$t0,%lo(bar)
  lw      $t0,%got(foo)($gp)
$LC0:
  nop

  .data
  .space 0x9000
$LC1:
  .word 0
  .space 0x18000
  .word 0
.global bar
.hidden bar
bar:
  .word 0
