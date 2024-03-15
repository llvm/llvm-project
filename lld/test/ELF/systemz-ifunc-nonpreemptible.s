# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x-none-linux-gnu %s -o %t.o
# RUN: ld.lld -static %t.o -o %t
# RUN: ld.lld -static %t.o -o %t.apply --apply-dynamic-relocs
# RUN: llvm-readelf --section-headers --relocations --symbols %t | FileCheck %s
# RUN: llvm-readelf -x .got.plt %t | FileCheck %s --check-prefix=NO-APPLY-RELOC
# RUN: llvm-readelf -x .got.plt %t.apply | FileCheck %s --check-prefix=APPLY-RELOC
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --check-prefix=DISASM

# CHECK:      Section Headers:
# CHECK-NEXT:  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK-NEXT:  [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
# CHECK-NEXT:  [ 1] .rela.dyn         RELA            0000000001000158 000158 000030 18  AI  0   4  8
# CHECK-NEXT:  [ 2] .text             PROGBITS        0000000001001188 000188 00001c 00  AX  0   0  4
# CHECK-NEXT:  [ 3] .iplt             PROGBITS        00000000010011b0 0001b0 000040 00  AX  0   0 16
# CHECK-NEXT:  [ 4] .got.plt          PROGBITS        00000000010021f0 0001f0 000010 00  WA  0   0  8

# CHECK:      Relocation section '.rela.dyn' at offset 0x158 contains 2 entries:
# CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# CHECK-NEXT: 00000000010021f0  000000000000003d R_390_IRELATIVE                   1001188
# CHECK-NEXT: 00000000010021f8  000000000000003d R_390_IRELATIVE                   100118a

# CHECK:      Symbol table '.symtab' contains 6 entries:
# CHECK-NEXT:   Num:    Value          Size Type    Bind   Vis       Ndx Name
# CHECK-NEXT:     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# CHECK-NEXT:     1: 0000000001000158     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_start
# CHECK-NEXT:     2: 0000000001000188     0 NOTYPE  LOCAL  HIDDEN      1 __rela_iplt_end
# CHECK-NEXT:     3: 0000000001001188     0 IFUNC   GLOBAL DEFAULT     2 foo
# CHECK-NEXT:     4: 000000000100118a     0 IFUNC   GLOBAL DEFAULT     2 bar
# CHECK-NEXT:     5: 000000000100118c     0 NOTYPE  GLOBAL DEFAULT     2 _start

# NO-APPLY-RELOC-LABEL:  Hex dump of section '.got.plt':
# NO-APPLY-RELOC-NEXT:     0x010021f0 00000000 00000000 00000000 00000000
# NO-APPLY-RELOC-EMPTY:

# APPLY-RELOC-LABEL:  Hex dump of section '.got.plt':
# APPLY-RELOC-NEXT:     0x010021f0 00000000 01001188 00000000 0100118a
# APPLY-RELOC-EMPTY:

# DISASM: Disassembly of section .text:
# DISASM: 0000000001001188 <foo>:
# DISASM-NEXT:  br      %r14
# DISASM: 000000000100118a <bar>:
# DISASM-NEXT:  br      %r14
# DISASM: 000000000100118c  <_start>:
# DISASM-NEXT:  brasl   %r14, 0x10011b0
# DISASM-NEXT:  brasl   %r14, 0x10011d0
# DISASM-NEXT:  larl    %r2, 0x1000158
# DISASM-NEXT:  larl    %r2, 0x1000188
# DISASM: Disassembly of section .iplt:
# DISASM: <.iplt>:
# DISASM:        10011b0:       larl    %r1, 0x10021f0
# DISASM-NEXT:   10011b6:       lg      %r1, 0(%r1)
# DISASM-NEXT:   10011bc:       br      %r1
# DISASM:        10011d0:       larl    %r1, 0x10021f8
# DISASM-NEXT:   10011d6:       lg      %r1, 0(%r1)
# DISASM-NEXT:   10011dc:       br      %r1

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 br %r14

.type bar STT_GNU_IFUNC
.globl bar
bar:
 br %r14

.globl _start
_start:
 brasl %r14, foo@plt
 brasl %r14, bar@plt
 larl %r2, __rela_iplt_start
 larl %r2, __rela_iplt_end
