// REQUIRES: x86

/// For non-preemptable ifunc, place ifunc PLT entries after regular PLT entries.

// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld --shared -o %t.so %t.o
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r %t.so | FileCheck %s

// Check that an IRELATIVE relocation is used for a non-preemptible ifunc
// handler and a JUMP_SLOT is used for a preemptible ifunc
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <fct>:
// DISASM-NEXT:     1308:       retq
// DISASM:     <fct2>:
// DISASM-NEXT:     1309:       retq
// DISASM:     <f1>:
// DISASM-NEXT:     130a:       callq   0x1350
// DISASM-NEXT:     130f:       callq   0x1330
// DISASM-NEXT:     1314:       callq   0x1340
// DISASM-NEXT:     1319:       retq
// DISASM:     <f2>:
// DISASM-NEXT:     131a:       retq
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: <.plt>:
// DISASM-NEXT:     1320:       pushq   8482(%rip)
// DISASM-NEXT:     1326:       jmpq    *8484(%rip)
// DISASM-NEXT:     132c:       nopl    (%rax)
// DISASM-EMPTY:
// DISASM-NEXT:   <fct2@plt>:
// DISASM-NEXT:     1330:       jmpq    *8482(%rip)
// DISASM-NEXT:     1336:       pushq   $0
// DISASM-NEXT:     133b:       jmp     0x1320 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT:   <f2@plt>:
// DISASM-NEXT:     1340:       jmpq    *8474(%rip)
// DISASM-NEXT:     1346:       pushq   $1
// DISASM-NEXT:     134b:       jmp     0x1320 <.plt>
// DISASM:      Disassembly of section .iplt:
// DISASM-EMPTY:
// DISASM:      <.iplt>:
// DISASM-NEXT:     1350:       jmpq    *8466(%rip)
// DISASM-NEXT:     1356:       pushq   $0
// DISASM-NEXT:     135b:       jmp     0x1320 <.plt>

// CHECK: Relocations [
// CHECK-NEXT:   Section (5) .rela.dyn {
// CHECK-NEXT:     0x3468 R_X86_64_IRELATIVE - 0x1308
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (6) .rela.plt {
// CHECK-NEXT:     0x3458 R_X86_64_JUMP_SLOT fct2 0x0
// CHECK-NEXT:     0x3460 R_X86_64_JUMP_SLOT f2 0x0
// CHECK-NEXT:   }

 // Hidden expect IRELATIVE
 .globl fct
 .hidden fct
 .type  fct, STT_GNU_IFUNC
fct:
 ret

 // Not hidden expect JUMP_SLOT
 .globl fct2
 .type  fct2, STT_GNU_IFUNC
fct2:
 ret

 .globl f1
 .type f1, @function
f1:
 call fct
 call fct2
 call f2@PLT
 ret

 .globl f2
 .type f2, @function
f2:
 ret
