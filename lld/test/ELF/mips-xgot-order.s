# REQUIRES: mips
# Check that GOT entries accessed via 16-bit indexing are allocated
# in the beginning of the GOT.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.exe | FileCheck %s
# RUN: llvm-readelf -s -A %t.exe | FileCheck -check-prefix=GOT %s

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <__start>:
# CHECK-NEXT:    lui     $2, 0
# CHECK-NEXT:    lw      $2, -32732($2)
# CHECK-NEXT:    lui     $2, 0
# CHECK-NEXT:    lw      $2, -32728($2)
#
# CHECK:      <bar>:
# CHECK-NEXT:    lw      $2, -32736($2)
# CHECK-NEXT:    lw      $2, -32744($2)
# CHECK-NEXT:    addi    $2, $2, {{.*}}

# GOT: Symbol table '.symtab'
# GOT:    Num:               Value  Size Type    Bind   Vis       Ndx Name
# GOT:           [[LOC:[0-9a-f]+]]     0 NOTYPE  LOCAL  DEFAULT     4 loc
# GOT:         [[START:[0-9a-f]+]]     0 NOTYPE  GLOBAL DEFAULT     3 __start
# GOT:           [[BAR:[0-9a-f]+]]     0 NOTYPE  GLOBAL DEFAULT     3 bar

# GOT:      Static GOT:
# GOT:       Local entries:
# GOT:         Address     Access  Initial
# GOT-NEXT:            -32744(gp) 00030000
# GOT-NEXT:            -32740(gp) 00040000
# GOT-NEXT:            -32736(gp) [[BAR]]
# GOT-NEXT:            -32732(gp) [[START]]
# GOT-NEXT:            -32728(gp) [[LOC]]

  .text
  .global __start, bar
__start:
  lui   $2, %got_hi(__start)
  lw    $2, %got_lo(__start)($2)
  lui   $2, %got_hi(loc)
  lw    $2, %got_lo(loc)($2)
bar:
  lw    $2, %got(bar)($2)
  lw    $2, %got(loc)($2)
  addi  $2, $2, %lo(loc)

  .data
loc:
  .word 0
