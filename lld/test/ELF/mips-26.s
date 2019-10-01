# REQUIRES: mips
# Check R_MIPS_26 relocation handling.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %S/Inputs/mips-dynamic.s -o %t2.o
# RUN: ld.lld %t2.o -shared -o %t.so
# RUN: ld.lld %t1.o %t.so -o %t.exe
# RUN: llvm-objdump -d --no-show-raw-insn %t.exe | FileCheck %s
# RUN: llvm-readobj --dynamic-table -S -r --mips-plt-got %t.exe \
# RUN:   | FileCheck -check-prefix=REL %s

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: bar:
# CHECK-NEXT:   20000:       jal     131096 <loc>
# CHECK-NEXT:   20004:       nop
#
# CHECK:      __start:
# CHECK-NEXT:   20008:       jal     131072 <bar>
# CHECK-NEXT:   2000c:       nop
# CHECK-NEXT:   20010:       jal     131136
#                                    ^-- 0x20040 gotplt[foo0]
# CHECK-NEXT:   20014:       nop
#
# CHECK:      loc:
# CHECK-NEXT:   20018:       nop
# CHECK-EMPTY:
# CHECK-NEXT: Disassembly of section .plt:
# CHECK-EMPTY:
# CHECK-NEXT: .plt:
# CHECK-NEXT:   20020:       lui     $gp, 3
# CHECK-NEXT:   20024:       lw      $25, 4($gp)
# CHECK-NEXT:   20028:       addiu   $gp, $gp, 4
# CHECK-NEXT:   2002c:       subu    $24, $24, $gp
# CHECK-NEXT:   20030:       move    $15, $ra
# CHECK-NEXT:   20034:       srl     $24, $24, 2
# CHECK-NEXT:   20038:       jalr    $25
# CHECK-NEXT:   2003c:       addiu   $24, $24, -2
# CHECK-NEXT:   20040:       lui     $15, 3
# CHECK-NEXT:   20044:       lw      $25, 12($15)
# CHECK-NEXT:   20048:       jr      $25
# CHECK-NEXT:   2004c:       addiu   $24, $15, 12

# REL:      Name: .plt
# REL-NEXT: Type: SHT_PROGBITS
# REL-NEXT: Flags [ (0x6)
# REL-NEXT:   SHF_ALLOC
# REL-NEXT:   SHF_EXECINSTR
# REL-NEXT: ]
# REL-NEXT: Address: 0x[[PLTADDR:[0-9A-F]+]]

# REL:      Name: .got.plt
# REL-NEXT: Type: SHT_PROGBITS
# REL-NEXT: Flags [ (0x3)
# REL-NEXT:   SHF_ALLOC
# REL-NEXT:   SHF_WRITE
# REL-NEXT: ]
# REL-NEXT: Address: 0x[[GOTPLTADDR:[0-9A-F]+]]

# REL: Relocations [
# REL-NEXT:   Section (7) .rel.plt {
# REL-NEXT:     0x[[PLTSLOT:[0-9A-F]+]] R_MIPS_JUMP_SLOT foo0 0x0
# REL-NEXT:   }
# REL-NEXT: ]

# REL: 0x70000032  MIPS_PLTGOT  0x[[GOTPLTADDR]]

# REL:      Primary GOT {
# REL:        Local entries [
# REL-NEXT:   ]
# REL-NEXT:   Global entries [
# REL-NEXT:   ]
# REL:      PLT GOT {
# REL:        Entries [
# REL-NEXT:     Entry {
# REL-NEXT:       Address: 0x[[PLTSLOT]]
# REL-NEXT:       Initial: 0x[[PLTADDR]]
# REL-NEXT:       Value: 0x0
# REL-NEXT:       Type: Function
# REL-NEXT:       Section: Undefined
# REL-NEXT:       Name: foo0
# REL-NEXT:     }
# REL-NEXT:   ]

  .text
  .globl bar
bar:
  jal loc         # R_MIPS_26 against .text + offset

  .globl __start
__start:
  jal bar         # R_MIPS_26 against global 'bar' from object file
  jal foo0        # R_MIPS_26 against 'foo0' from DSO

loc:
  nop
