// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s -o %t.o
// RUN: ld.lld -static %t.o -o %tout
// RUN: llvm-objdump -d %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r -symbols -sections %tout | FileCheck %s
// REQUIRES: aarch64

// CHECK:      Sections [
// CHECK:       Section {
// CHECK:       Index: 1
// CHECK-NEXT:  Name: .rela.plt
// CHECK-NEXT:  Type: SHT_RELA
// CHECK-NEXT:  Flags [
// CHECK-NEXT:    SHF_ALLOC
// CHECK-NEXT:  ]
// CHECK-NEXT:  Address: [[RELA:.*]]
// CHECK-NEXT:  Offset: 0x158
// CHECK-NEXT:  Size: 48
// CHECK-NEXT:  Link: 6
// CHECK-NEXT:  Info: 0
// CHECK-NEXT:  AddressAlignment: 8
// CHECK-NEXT:  EntrySize: 24
// CHECK-NEXT: }
// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.plt {
// CHECK-NEXT:     0x30018 R_AARCH64_IRELATIVE
// CHECK-NEXT:     0x30020 R_AARCH64_IRELATIVE
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK:      Symbols [
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name:
// CHECK-NEXT:    Value: 0x0
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: Undefined
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: $x.0
// CHECK-NEXT:    Value: 0x20000
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: __rela_iplt_end
// CHECK-NEXT:    Value: 0x10188
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .rela.plt
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: __rela_iplt_start
// CHECK-NEXT:    Value: 0x10158
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .rela.plt
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: _start
// CHECK-NEXT:    Value: 0x20008
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: bar
// CHECK-NEXT:    Value: 0x20004
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: foo
// CHECK-NEXT:    Value: 0x20000
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT: ]

// 344 = 0x158
// 392 = 0x188
// DISASM:      Disassembly of section .text:
// DISASM-NEXT: foo:
// DISASM-NEXT:  20000: c0 03 5f d6 ret
// DISASM:      bar:
// DISASM-NEXT:  20004: c0 03 5f d6 ret
// DISASM:      _start:
// DISASM-NEXT:  20008: 0e 00 00 94 bl #56
// DISASM-NEXT:  2000c: 11 00 00 94 bl #68
// DISASM-NEXT:  20010: 42 60 05 91 add x2, x2, #344
// DISASM-NEXT:  20014: 42 20 06 91 add x2, x2, #392
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:  20020: f0 7b bf a9 stp x16, x30, [sp, #-16]!
// DISASM-NEXT:  20024: 90 00 00 90 adrp x16, #65536
// DISASM-NEXT:  20028: 11 0a 40 f9 ldr x17, [x16, #16]
// DISASM-NEXT:  2002c: 10 42 00 91 add x16, x16, #16
// DISASM-NEXT:  20030: 20 02 1f d6 br x17
// DISASM-NEXT:  20034: 1f 20 03 d5 nop
// DISASM-NEXT:  20038: 1f 20 03 d5 nop
// DISASM-NEXT:  2003c: 1f 20 03 d5 nop
// DISASM-NEXT:  20040: 90 00 00 90 adrp x16, #65536
// DISASM-NEXT:  20044: 11 0e 40 f9 ldr x17, [x16, #24]
// DISASM-NEXT:  20048: 10 62 00 91 add x16, x16, #24
// DISASM-NEXT:  2004c: 20 02 1f d6 br x17
// DISASM-NEXT:  20050: 90 00 00 90 adrp x16, #65536
// DISASM-NEXT:  20054: 11 12 40 f9 ldr x17, [x16, #32]
// DISASM-NEXT:  20058: 10 82 00 91 add x16, x16, #32
// DISASM-NEXT:  2005c: 20 02 1f d6 br x17

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 ret

.type bar STT_GNU_IFUNC
.globl bar
bar:
 ret

.globl _start
_start:
 bl foo
 bl bar
 add x2, x2, :lo12:__rela_iplt_start
 add x2, x2, :lo12:__rela_iplt_end
