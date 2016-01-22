# Check R_MIPS_GOT16 relocation calculation.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-objdump -d -t %t.so | FileCheck %s
# RUN: llvm-readobj -r -mips-plt-got %t.so | FileCheck -check-prefix=GOT %s

# REQUIRES: mips

# CHECK:      Disassembly of section .text:
# CHECK-NEXT: __start:
# CHECK-NEXT:    10000:       8f 88 80 18     lw      $8, -32744($gp)
# CHECK-NEXT:    10004:       21 08 00 1c     addi    $8, $8, 28
# CHECK-NEXT:    10008:       8f 88 80 1c     lw      $8, -32740($gp)
# CHECK-NEXT:    1000c:       21 08 00 00     addi    $8, $8, 0
# CHECK-NEXT:    10010:       8f 88 80 20     lw      $8, -32736($gp)
# CHECK-NEXT:    10014:       21 08 00 04     addi    $8, $8, 4
# CHECK-NEXT:    10018:       8f 88 80 24     lw      $8, -32732($gp)
#
# CHECK: SYMBOL TABLE:
# CHECK: 0001001c         .text           00000000 $LC0
# CHECK: 00030000         .data           00000000 $LC1
# CHECK: 00030004         .data           00000000 .hidden bar
# CHECK: 00000000         *UND*           00000000 foo

# GOT:      Relocations [
# GOT-NEXT: ]

# GOT:      Primary GOT {
# GOT-NEXT:   Canonical gp value: 0x27FF0
# GOT-NEXT:   Reserved entries [
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address: 0x20000
# GOT-NEXT:       Access: -32752
# GOT-NEXT:       Initial: 0x0
# GOT-NEXT:       Purpose: Lazy resolver
# GOT-NEXT:     }
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address: 0x20004
# GOT-NEXT:       Access: -32748
# GOT-NEXT:       Initial: 0x80000000
# GOT-NEXT:       Purpose: Module pointer (GNU extension)
# GOT-NEXT:     }
# GOT-NEXT:   ]
# GOT-NEXT:   Local entries [
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address: 0x20008
# GOT-NEXT:       Access: -32744
# GOT-NEXT:       Initial: 0x10000
# GOT-NEXT:     }
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address: 0x2000C
# GOT-NEXT:       Access: -32740
# GOT-NEXT:       Initial: 0x30000
# GOT-NEXT:     }
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address: 0x20010
# GOT-NEXT:       Access: -32736
# GOT-NEXT:       Initial: 0x30004
# GOT-NEXT:     }
# GOT-NEXT:   ]
# GOT-NEXT:   Global entries [
# GOT-NEXT:     Entry {
# GOT-NEXT:       Address: 0x20014
# GOT-NEXT:       Access: -32732
# GOT-NEXT:       Initial: 0x0
# GOT-NEXT:       Value: 0x0
# GOT-NEXT:       Type: None
# GOT-NEXT:       Section: Undefined
# GOT-NEXT:       Name: foo@
# GOT-NEXT:     }
# GOT-NEXT:   ]
# GOT-NEXT:   Number of TLS and multi-GOT entries: 0
# GOT-NEXT: }

  .text
  .globl  __start
__start:
  lw      $t0,%got($LC0)($gp)
  addi    $t0,$t0,%lo($LC0)
  lw      $t0,%got($LC1)($gp)
  addi    $t0,$t0,%lo($LC1)
  lw      $t0,%got(bar)($gp)
  addi    $t0,$t0,%lo(bar)
  lw      $t0,%got(foo)($gp)
$LC0:
  nop

  .data
$LC1:
  .word 0
.global bar
.hidden bar
bar:
  .word 0
