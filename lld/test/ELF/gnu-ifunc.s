// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld -static %t.o -o %tout
// RUN: llvm-objdump -d %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r --symbols --sections %tout | FileCheck %s

// CHECK:      Sections [
// CHECK:       Section {
// CHECK:       Index: 1
// CHECK-NEXT:  Name: .rela.dyn
// CHECK-NEXT:  Type: SHT_RELA
// CHECK-NEXT:  Flags [
// CHECK-NEXT:    SHF_ALLOC
// CHECK-NEXT:    SHF_INFO_LINK
// CHECK-NEXT:  ]
// CHECK-NEXT:  Address: [[RELA:.*]]
// CHECK-NEXT:  Offset: 0x158
// CHECK-NEXT:  Size: 48
// CHECK-NEXT:  Link: 0
// CHECK-NEXT:  Info: [[GOTPLT:.*]]
// CHECK-NEXT:  AddressAlignment: 8
// CHECK-NEXT:  EntrySize: 24
// CHECK-NEXT: }
// CHECK:      Index: [[GOTPLT]]
// CHECK-NEXT: Name: .got.plt
// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x2021D0 R_X86_64_IRELATIVE
// CHECK-NEXT:     0x2021D8 R_X86_64_IRELATIVE
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
// CHECK-NEXT:    Name: __rela_iplt_start
// CHECK-NEXT:    Value: [[RELA]]
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .rela.dyn
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: __rela_iplt_end
// CHECK-NEXT:    Value: 0x200188
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Local
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other [
// CHECK-NEXT:      STV_HIDDEN
// CHECK-NEXT:    ]
// CHECK-NEXT:    Section: .rela.dyn
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: foo
// CHECK-NEXT:    Value: 0x201188
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: bar
// CHECK-NEXT:    Value: 0x201189
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: GNU_IFunc
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT:  Symbol {
// CHECK-NEXT:    Name: _start
// CHECK-NEXT:    Value: 0x20118A
// CHECK-NEXT:    Size: 0
// CHECK-NEXT:    Binding: Global
// CHECK-NEXT:    Type: None
// CHECK-NEXT:    Other: 0
// CHECK-NEXT:    Section: .text
// CHECK-NEXT:  }
// CHECK-NEXT: ]

// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <foo>:
// DISASM-NEXT:  201188: {{.*}} retq
// DISASM:      <bar>:
// DISASM-NEXT:  201189: {{.*}} retq
// DISASM:      <_start>:
// DISASM-NEXT:  20118a: {{.*}} callq 0x2011b0
// DISASM-NEXT:  20118f: {{.*}} callq 0x2011c0
// DISASM-NEXT:  201194: {{.*}} movl $2097496, %edx
// DISASM-NEXT:  201199: {{.*}} movl $2097544, %edx
// DISASM-NEXT:  20119e: {{.*}} movl $2097545, %edx
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .iplt:
// DISASM-EMPTY:
// DISASM-NEXT: <.iplt>:
// DISASM-NEXT:  2011b0: {{.*}} jmpq *4122(%rip)
// DISASM-NEXT:  2011b6: {{.*}} pushq $0
// DISASM-NEXT:  2011bb: {{.*}} jmp 0x0
// DISASM-NEXT:  2011c0: {{.*}} jmpq *4114(%rip)
// DISASM-NEXT:  2011c6: {{.*}} pushq $1
// DISASM-NEXT:  2011cb: {{.*}} jmp 0x0

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
 call foo
 call bar
 movl $__rela_iplt_start,%edx
 movl $__rela_iplt_end,%edx
 movl $__rela_iplt_end + 1,%edx
