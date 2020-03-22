// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared -soname=so %t2.o -o %t2.so
// RUN: ld.lld -shared %t.o %t2.so -o %t
// RUN: ld.lld %t.o %t2.so -o %t3
// RUN: llvm-readobj -S -r %t | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASM %s
// RUN: llvm-readobj -S -r %t3 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump -d --no-show-raw-insn %t3 | FileCheck --check-prefix=DISASM2 %s

// CHECK:      Name: .plt
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_EXECINSTR
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x1320
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 64
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 16

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.plt {
// CHECK-NEXT:     0x3438 R_X86_64_JUMP_SLOT bar 0x0
// CHECK-NEXT:     0x3440 R_X86_64_JUMP_SLOT zed 0x0
// CHECK-NEXT:     0x3448 R_X86_64_JUMP_SLOT _start 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK2:      Name: .plt
// CHECK2-NEXT: Type: SHT_PROGBITS
// CHECK2-NEXT: Flags [
// CHECK2-NEXT:   SHF_ALLOC
// CHECK2-NEXT:   SHF_EXECINSTR
// CHECK2-NEXT: ]
// CHECK2-NEXT: Address: 0x2012E0
// CHECK2-NEXT: Offset:
// CHECK2-NEXT: Size: 48
// CHECK2-NEXT: Link: 0
// CHECK2-NEXT: Info: 0
// CHECK2-NEXT: AddressAlignment: 16

// CHECK2:      Relocations [
// CHECK2-NEXT:   Section ({{.*}}) .rela.plt {
// CHECK2-NEXT:     0x2033F8 R_X86_64_JUMP_SLOT bar 0x0
// CHECK2-NEXT:     0x203400 R_X86_64_JUMP_SLOT zed 0x0
// CHECK2-NEXT:   }
// CHECK2-NEXT: ]

// DISASM:      <_start>:
// DISASM-NEXT:   jmp  {{.*}} <bar@plt>
// DISASM-NEXT:   jmp  {{.*}} <bar@plt>
// DISASM-NEXT:   jmp  {{.*}} <zed@plt>
// DISASM-NEXT:   jmp  {{.*}} <_start@plt>

// 0x3018 - 0x1036  = 8162
// 0x3020 - 0x1046  = 8154
// 0x3028 - 0x1056  = 8146

// DISASM:      Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: <.plt>:
// DISASM-NEXT:   1320:       pushq 8450(%rip)
// DISASM-NEXT:               jmpq *8452(%rip)
// DISASM-NEXT:               nopl (%rax)
// DISASM-EMPTY:
// DISASM-NEXT:   <bar@plt>:
// DISASM-NEXT:   1330:       jmpq *8450(%rip)
// DISASM-NEXT:               pushq $0
// DISASM-NEXT:               jmp 0x1320 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT:   <zed@plt>:
// DISASM-NEXT:   1340:       jmpq *8442(%rip)
// DISASM-NEXT:               pushq $1
// DISASM-NEXT:               jmp 0x1320 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT:   <_start@plt>:
// DISASM-NEXT:   1350:       jmpq *8434(%rip)
// DISASM-NEXT:               pushq $2
// DISASM-NEXT:               jmp 0x1320 <.plt>

// 0x201030 - (0x201000 + 1) - 4 = 43
// 0x201030 - (0x201005 + 1) - 4 = 38
// 0x201040 - (0x20100a + 1) - 4 = 49
// 0x201000 - (0x20100f + 1) - 4 = -20

// DISASM2:      <_start>:
// DISASM2-NEXT:   jmp  0x2012f0 <bar@plt>
// DISASM2-NEXT:   jmp  0x2012f0 <bar@plt>
// DISASM2-NEXT:   jmp  0x201300 <zed@plt>
// DISASM2-NEXT:   jmp  0x2012c0 <_start>

// 0x202018 - 0x201036  = 4066
// 0x202020 - 0x201046  = 4058

// DISASM2:      Disassembly of section .plt:
// DISASM2-EMPTY:
// DISASM2-NEXT: <.plt>:
// DISASM2-NEXT:  2012e0:       pushq 8450(%rip)
// DISASM2-NEXT:                jmpq *8452(%rip)
// DISASM2-NEXT:                nopl  (%rax)
// DISASM2-EMPTY:
// DISASM2-NEXT: <bar@plt>:
// DISASM2-NEXT:  2012f0:       jmpq *8450(%rip)
// DISASM2-NEXT:                pushq $0
// DISASM2-NEXT:                jmp 0x2012e0 <.plt>
// DISASM2-EMPTY:
// DISASM2-NEXT: <zed@plt>:
// DISASM2-NEXT:  201300:       jmpq *8442(%rip)
// DISASM2-NEXT:                pushq $1
// DISASM2-NEXT:                jmp 0x2012e0 <.plt>
// DISASM2-NOT:   {{.}}

.global _start
_start:
  jmp bar@PLT
  jmp bar@PLT
  jmp zed@PLT
  jmp _start@plt
