// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld --hash-style=sysv -shared %t -o %tout
// RUN: llvm-readobj --sections -r %tout | FileCheck %s
// RUN: llvm-objdump -d %tout | FileCheck %s --check-prefix=DIS

  leaq  a@tlsld(%rip), %rdi
  callq __tls_get_addr@PLT
  leaq  b@tlsld(%rip), %rdi
  callq __tls_get_addr@PLT
  leaq  a@dtpoff(%rax), %rcx
  leaq  b@dtpoff(%rax), %rcx
  .long b@dtpoff, 0
  leaq  c@tlsgd(%rip), %rdi
  rex64
  callq __tls_get_addr@PLT
  leaq  a@dtpoff(%rax), %rcx
  // Initial Exec Model Code Sequence, II
  movq c@gottpoff(%rip),%rax
  movq %fs:(%rax),%rax
  movabs $a@dtpoff, %rax
  movabs $b@dtpoff, %rax
  movabs $a@dtpoff, %rax

  .global a
  .hidden a
  .section .tbss,"awT",@nobits
  .align 4
a:
  .long 0

  .section .tbss,"awT",@nobits
  .align 4
b:
  .long 0
  .global c
  .section .tbss,"awT",@nobits
  .align 4
c:
  .long 0

// Get the address of the got, and check that it has 4 entries.

// CHECK:      Sections [
// CHECK:          Name: .got (
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x24A0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 40

// CHECK:      Relocations [
// CHECK:        Section ({{.+}}) .rela.dyn {
// CHECK-NEXT:     0x24A0 R_X86_64_DTPMOD64 - 0x0
// CHECK-NEXT:     0x24B0 R_X86_64_DTPMOD64 c 0x0
// CHECK-NEXT:     0x24B8 R_X86_64_DTPOFF64 c 0x0
// CHECK-NEXT:     0x24C0 R_X86_64_TPOFF64 c 0x0
// CHECK-NEXT:   }

// 4457 = (0x24A0 + -4) - (0x1330 + 3) // PC relative offset to got entry.
// 4445 = (0x24B0 + -4) - (0x133c + 3) // PC relative offset to got entry.
// 4427 = (0x24B8 + -4) - (0x135e + 3) // PC relative offset to got entry.
// 4423 = (0x24C0 + -4) - (0x1372 + 3) // PC relative offset to got entry.

// DIS:      Disassembly of section .text:
// DIS-EMPTY:
// DIS-NEXT: .text:
// DIS-NEXT:     1330: {{.+}} leaq    4457(%rip), %rdi
// DIS-NEXT:           {{.+}} callq
// DIS-NEXT:     133c: {{.+}} leaq    4445(%rip), %rdi
// DIS-NEXT:           {{.+}} callq
// DIS-NEXT:           {{.+}} leaq    (%rax), %rcx
// DIS-NEXT:           {{.+}} leaq    4(%rax), %rcx
// DIS-NEXT:           04 00
// DIS-NEXT:           00 00
// DIS-NEXT:           00 00
// DIS-NEXT:           00 00
// DIS-NEXT:     135e: {{.+}} leaq    4427(%rip), %rdi
// DIS-NEXT:           {{.+}} callq
// DIS-NEXT:           {{.+}} leaq    (%rax), %rcx
// DIS-NEXT:     1372: {{.+}} movq    4423(%rip), %rax
// DIS-NEXT:           {{.+}} movq    %fs:(%rax), %rax
// DIS-NEXT:           {{.+}} movabsq $0, %rax
// DIS-NEXT:           {{.+}} movabsq $4, %rax
// DIS-NEXT:           {{.+}} movabsq $0, %rax
