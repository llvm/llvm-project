# REQUIRES: mips
# Check R_MIPS_26 relocation handling.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %S/Inputs/mips-dynamic.s -o %t2.o
# RUN: ld.lld %t2.o -shared -o %t.so
# RUN: ld.lld %t1.o %t.so -o %t.exe
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.exe | FileCheck %s
# RUN: llvm-readobj --dynamic-table -S -r -A %t.exe \
# RUN:   | FileCheck -check-prefix=REL %s

# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: bar:
# CHECK-NEXT:   [[BAR:[0-9a-f]+]]:  jal  0x[[LOC:[0-9a-f]+]] <loc>
# CHECK-NEXT:   {{.*}}:              nop
#
# CHECK:      __start:
# CHECK-NEXT:   {{.*}}:       jal     0x[[BAR]] <bar>
# CHECK-NEXT:   {{.*}}:       nop
# CHECK-NEXT:   {{.*}}:       jal     0x[[FOO0:[0-9a-f]+]]
#                                     ^-- gotplt[foo0]
# CHECK-NEXT:   {{.*}}:       nop
#
# CHECK:      loc:
# CHECK-NEXT:   [[LOC]]:      nop
# CHECK-EMPTY:
# CHECK-NEXT: Disassembly of section .plt:
# CHECK-EMPTY:
# CHECK-NEXT: .plt:
# CHECK-NEXT:   {{.*}}:       lui     $gp, 0x3
# CHECK-NEXT:   {{.*}}:       lw      $25, {{.*}}($gp)
# CHECK-NEXT:   {{.*}}:       addiu   $gp, $gp, {{.*}}
# CHECK-NEXT:   {{.*}}:       subu    $24, $24, $gp
# CHECK-NEXT:   {{.*}}:       move    $15, $ra
# CHECK-NEXT:   {{.*}}:       srl     $24, $24, 0x2
# CHECK-NEXT:   {{.*}}:       jalr    $25
# CHECK-NEXT:   {{.*}}:       addiu   $24, $24, -0x2
# CHECK-NEXT:   [[FOO0]]:     lui     $15, 0x3
# CHECK-NEXT:   {{.*}}:       lw      $25, {{.*}}($15)
# CHECK-NEXT:   {{.*}}:       jr      $25
# CHECK-NEXT:   {{.*}}:       addiu   $24, $15, {{.*}}

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
