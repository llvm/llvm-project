// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld -static %t.o -o %tout
// RUN: llvm-objdump -d %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r -symbols -sections %tout | FileCheck %s --check-prefix=CHECK
// REQUIRES: x86

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
// CHECK-NEXT:  Link: 5
// CHECK-NEXT:  Info: 0
// CHECK-NEXT:  AddressAlignment: 8
// CHECK-NEXT:  EntrySize: 24
// CHECK-NEXT: }
// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.plt {
// CHECK-NEXT:     0x12018 R_X86_64_IRELATIVE
// CHECK-NEXT:     0x12020 R_X86_64_IRELATIVE
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
// CHECK-NEXT:    Name: __rela_iplt_end
// CHECK-NEXT:    Value: 0x10188
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: Absolute
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: __rela_iplt_start
// CHECK-NEXT:    Value: [[RELA]]
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: Absolute
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: _start
// CHECK-NEXT:    Value: 0x11002
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: bar
// CHECK-NEXT:    Value: 0x11001
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: foo
// CHECK-NEXT:    Value: 0x11000
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT: ]

// DISASM:      Disassembly of section .text:
// DISASM-NEXT: foo:
// DISASM-NEXT:    11000: c3 retq
// DISASM:      bar:
// DISASM-NEXT:    11001: c3 retq
// DISASM:      _start:
// DISASM-NEXT:    11002: e8 29 00 00 00 callq 41
// DISASM-NEXT:    11007: e8 34 00 00 00 callq 52
// DISASM-NEXT:    1100c: ba 58 01 01 00 movl $65880, %edx
// DISASM-NEXT:    11011: ba 88 01 01 00 movl $65928, %edx
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:    11020: ff 35 e2 0f 00 00 pushq 4066(%rip)
// DISASM-NEXT:    11026: ff 25 e4 0f 00 00 jmpq *4068(%rip)
// DISASM-NEXT:    1102c: 0f 1f 40 00       nopl (%rax)
// DISASM-NEXT:    11030: ff 25 e2 0f 00 00 jmpq *4066(%rip)
// DISASM-NEXT:    11036: 68 00 00 00 00    pushq $0
// DISASM-NEXT:    1103b: e9 e0 ff ff ff    jmp -32
// DISASM-NEXT:    11040: ff 25 da 0f 00 00 jmpq *4058(%rip)
// DISASM-NEXT:    11046: 68 01 00 00 00    pushq $1
// DISASM-NEXT:    1104b: e9 d0 ff ff ff    jmp -48

.text
.type foo STT_GNU_IFUNC
.globl foo
.type foo, @function
foo:
 ret

.type bar STT_GNU_IFUNC
.globl bar
.type bar, @function
bar:
 ret

.globl _start
_start:
 call foo
 call bar
 movl $__rela_iplt_start,%edx
 movl $__rela_iplt_end,%edx
