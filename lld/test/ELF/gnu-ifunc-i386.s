// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: ld.lld -static %t.o -o %tout
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r --symbols --sections %tout | FileCheck %s

// CHECK:      Sections [
// CHECK:       Section {
// CHECK:       Index: 1
// CHECK-NEXT:  Name: .rel.dyn
// CHECK-NEXT:  Type: SHT_REL
// CHECK-NEXT:  Flags [
// CHECK-NEXT:    SHF_ALLOC
// CHECK-NEXT:    SHF_INFO_LINK
// CHECK-NEXT:  ]
// CHECK-NEXT:  Address: [[RELA:.*]]
// CHECK-NEXT:  Offset: 0xD4
// CHECK-NEXT:  Size: 16
// CHECK-NEXT:  Link: 0
// CHECK-NEXT:  Info: 4
// CHECK-NEXT:  AddressAlignment: 4
// CHECK-NEXT:  EntrySize: 8
// CHECK-NEXT: }
// CHECK:     Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.dyn {
// CHECK-NEXT:     0x402120 R_386_IRELATIVE
// CHECK-NEXT:     0x402124 R_386_IRELATIVE
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK:      Symbols [
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name:
// CHECK-NEXT:   Value: 0x0
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: Undefined
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: __rel_iplt_start
// CHECK-NEXT:   Value: [[RELA]]
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other [
// CHECK-NEXT:     STV_HIDDEN
// CHECK-NEXT:   ]
// CHECK-NEXT:   Section: .rel.dyn
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: __rel_iplt_end
// CHECK-NEXT:   Value: 0x4000E4
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Local
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other [
// CHECK-NEXT:     STV_HIDDEN
// CHECK-NEXT:   ]
// CHECK-NEXT:   Section: .rel.dyn
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: bar
// CHECK-NEXT:   Value: 0x401100
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Global
// CHECK-NEXT:   Type: Function
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .iplt
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: bar_resolver
// CHECK-NEXT:   Value: 0x4010E4
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Global
// CHECK-NEXT:   Type: Function
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .text
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: foo
// CHECK-NEXT:   Value: 0x401110
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Global
// CHECK-NEXT:   Type: Function
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .iplt
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: foo_resolver
// CHECK-NEXT:   Value: 0x4010E5
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Global
// CHECK-NEXT:   Type: Function
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .text
// CHECK-NEXT: }
// CHECK-NEXT: Symbol {
// CHECK-NEXT:   Name: _start
// CHECK-NEXT:   Value: 0x4010E6
// CHECK-NEXT:   Size: 0
// CHECK-NEXT:   Binding: Global
// CHECK-NEXT:   Type: None
// CHECK-NEXT:   Other: 0
// CHECK-NEXT:   Section: .text
// CHECK-NEXT: }
// CHECK-NEXT:]

// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <bar_resolver>:
// DISASM-NEXT:   4010e4:       retl
// DISASM:      <foo_resolver>:
// DISASM-NEXT:   4010e5:       retl
// DISASM:      <_start>:
// DISASM-NEXT:   4010e6:       calll 0x401110 <foo>
// DISASM-NEXT:                 calll 0x401100 <bar>
// DISASM-NEXT:                 movl $4194516, %edx
// DISASM-NEXT:                 movl $4194532, %edx
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .iplt:
// DISASM-EMPTY:
// DISASM-NEXT: <bar>:
// DISASM-NEXT:   401100:       jmpl *4202784
// DISASM-NEXT:                 pushl $0
// DISASM-NEXT:                 jmp 0x0
// DISASM:      <foo>:
// DISASM-NEXT:   401110:       jmpl *4202788
// DISASM-NEXT:                 pushl $8
// DISASM-NEXT:                 jmp 0x0

.type bar STT_GNU_IFUNC
.globl bar
bar:
.type bar_resolver STT_FUNC
.globl bar_resolver
bar_resolver:
 ret

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
.type foo_resolver STT_FUNC
.globl foo_resolver
foo_resolver:
 ret

.globl _start
_start:
 call foo
 call bar
 movl $__rel_iplt_start,%edx
 movl $__rel_iplt_end,%edx
