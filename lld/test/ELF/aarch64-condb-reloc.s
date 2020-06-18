# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %p/Inputs/aarch64-condb-reloc.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %t2.o
# RUN: ld.lld %t1.o %t2.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
# RUN: ld.lld -shared %t1.o %t2.o -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=DSO %s
# RUN: llvm-readobj -S -r %t.so | FileCheck -check-prefix=DSOREL %s

# 0x11024 - 36 = 0x11000
# 0x11028 - 24 = 0x11010
# 0x1102c - 16 = 0x1101c
# CHECK:      Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <_foo>:
# CHECK-NEXT:    210120: nop
# CHECK-NEXT:    210124: nop
# CHECK-NEXT:    210128: nop
# CHECK-NEXT:    21012c: nop
# CHECK:      <_bar>:
# CHECK-NEXT:    210130: nop
# CHECK-NEXT:    210134: nop
# CHECK-NEXT:    210138: nop
# CHECK:      <_dah>:
# CHECK-NEXT:    21013c: nop
# CHECK-NEXT:    210140: nop
# CHECK:      <_start>:
# CHECK-NEXT:    210144: b.eq 0x210120 <_foo>
# CHECK-NEXT:    210148: b.eq 0x210130 <_bar>
# CHECK-NEXT:    21014c: b.eq 0x21013c <_dah>

#DSOREL:      Section {
#DSOREL:        Index:
#DSOREL:        Name: .got.plt
#DSOREL-NEXT:   Type: SHT_PROGBITS
#DSOREL-NEXT:   Flags [
#DSOREL-NEXT:     SHF_ALLOC
#DSOREL-NEXT:     SHF_WRITE
#DSOREL-NEXT:   ]
#DSOREL-NEXT:   Address: 0x30470
#DSOREL-NEXT:   Offset: 0x470
#DSOREL-NEXT:   Size: 48
#DSOREL-NEXT:   Link: 0
#DSOREL-NEXT:   Info: 0
#DSOREL-NEXT:   AddressAlignment: 8
#DSOREL-NEXT:   EntrySize: 0
#DSOREL-NEXT:  }
#DSOREL:      Relocations [
#DSOREL-NEXT:  Section ({{.*}}) .rela.plt {
#DSOREL-NEXT:    0x30488 R_AARCH64_JUMP_SLOT _foo
#DSOREL-NEXT:    0x30490 R_AARCH64_JUMP_SLOT _bar
#DSOREL-NEXT:    0x30498 R_AARCH64_JUMP_SLOT _dah
#DSOREL-NEXT:  }
#DSOREL-NEXT:]

#DSO:      Disassembly of section .text:
#DSO-EMPTY:
#DSO-NEXT: <_foo>:
#DSO-NEXT:     10338: nop
#DSO-NEXT:     1033c: nop
#DSO-NEXT:     10340: nop
#DSO-NEXT:     10344: nop
#DSO:      <_bar>:
#DSO-NEXT:     10348: nop
#DSO-NEXT:     1034c: nop
#DSO-NEXT:     10350: nop
#DSO:      <_dah>:
#DSO-NEXT:     10354: nop
#DSO-NEXT:     10358: nop
#DSO:      <_start>:
#DSO-NEXT:     1035c: b.eq 0x10390 <_foo@plt>
#DSO-NEXT:     10360: b.eq 0x103a0 <_bar@plt>
#DSO-NEXT:     10364: b.eq 0x103b0 <_dah@plt>
#DSO-EMPTY:
#DSO-NEXT: Disassembly of section .plt:
#DSO-EMPTY:
#DSO-NEXT: <.plt>:
#DSO-NEXT:     10370: stp x16, x30, [sp, #-16]!
#DSO-NEXT:     10374: adrp x16, #131072
#DSO-NEXT:     10378: ldr x17, [x16, #1152]
#DSO-NEXT:     1037c: add x16, x16, #1152
#DSO-NEXT:     10380: br x17
#DSO-NEXT:     10384: nop
#DSO-NEXT:     10388: nop
#DSO-NEXT:     1038c: nop
#DSO-EMPTY:
#DSO-NEXT:   <_foo@plt>:
#DSO-NEXT:     10390: adrp x16, #131072
#DSO-NEXT:     10394: ldr x17, [x16, #1160]
#DSO-NEXT:     10398: add x16, x16, #1160
#DSO-NEXT:     1039c: br x17
#DSO-EMPTY:
#DSO-NEXT:   <_bar@plt>:
#DSO-NEXT:     103a0: adrp x16, #131072
#DSO-NEXT:     103a4: ldr x17, [x16, #1168]
#DSO-NEXT:     103a8: add x16, x16, #1168
#DSO-NEXT:     103ac: br x17
#DSO-EMPTY:
#DSO-NEXT:   <_dah@plt>:
#DSO-NEXT:     103b0: adrp x16, #131072
#DSO-NEXT:     103b4: ldr x17, [x16, #1176]
#DSO-NEXT:     103b8: add x16, x16, #1176
#DSO-NEXT:     103bc: br x17

.globl _start
_start:
 b.eq _foo
 b.eq _bar
 b.eq _dah
