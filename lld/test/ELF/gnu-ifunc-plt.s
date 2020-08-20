// REQUIRES: x86

/// For non-preemptable ifunc, place ifunc PLT entries to the .iplt section.

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %S/Inputs/shared2-x86-64.s -o %t1.o
// RUN: ld.lld %t1.o --shared -soname=so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld --hash-style=sysv %t.so %t.o -o %tout
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -s %tout | FileCheck %s --check-prefix=GOTPLT
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

/// Check that the PLTRELSZ tag does not include the IRELATIVE relocations
// CHECK: DynamicSection [
// CHECK:   0x0000000000000008 RELASZ               72 (bytes)
// CHECK:   0x0000000000000002 PLTRELSZ             48 (bytes)

/// Check that the IRELATIVE relocations are placed to the .rela.dyn section after
/// other regular relocations (e.g. GLOB_DAT).
// CHECK:      Relocations [
// CHECK-NEXT:   Section (4) .rela.dyn {
// CHECK-NEXT:     0x202480 R_X86_64_GLOB_DAT bar3 0x0
// CHECK-NEXT:     0x2034B0 R_X86_64_IRELATIVE - 0x201318
// CHECK-NEXT:     0x2034B8 R_X86_64_IRELATIVE - 0x201319
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rela.plt {
// CHECK-NEXT:     0x2034A0 R_X86_64_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x2034A8 R_X86_64_JUMP_SLOT zed2 0x0
// CHECK-NEXT:   }

/// Check that .got.plt entries point back to PLT header
// GOTPLT: Contents of section .got.plt:
// GOTPLT-NEXT:  203488 90232000 00000000 00000000 00000000
// GOTPLT-NEXT:  203498 00000000 00000000 56132000 00000000
// GOTPLT-NEXT:  2034a8 66132000 00000000 00000000 00000000
// GOTPLT-NEXT:  2034b8 00000000 00000000

/// Check that we have 2 PLT sections: one regular .plt section and one
/// .iplt section for ifunc entries.
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <foo>:
// DISASM-NEXT:   201318:       retq
// DISASM:      <bar>:
// DISASM-NEXT:   201319:       retq
// DISASM:      <_start>:
// DISASM-NEXT:   20131a:       callq   0x201370
// DISASM-NEXT:   20131f:       callq   0x201380
// DISASM-NEXT:                 callq   {{.*}} <bar2@plt>
// DISASM-NEXT:                 callq   {{.*}} <zed2@plt>
// DISASM-NEXT:                 jmpq    *0x114c(%rip)
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: <.plt>:
// DISASM-NEXT:   201340:       pushq   0x214a(%rip)
// DISASM-NEXT:   201346:       jmpq    *0x214c(%rip)
// DISASM-NEXT:   20134c:       nopl    (%rax)
// DISASM-EMPTY:
// DISASM-NEXT:   <bar2@plt>:
// DISASM-NEXT:   201350:       jmpq    *0x214a(%rip)
// DISASM-NEXT:   201356:       pushq   $0x0
// DISASM-NEXT:   20135b:       jmp     0x201340 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT:   <zed2@plt>:
// DISASM-NEXT:   201360:       jmpq    *0x2142(%rip)
// DISASM-NEXT:   201366:       pushq   $0x1
// DISASM-NEXT:   20136b:       jmp     0x201340 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .iplt:
// DISASM-EMPTY:
// DISASM-NEXT: <.iplt>:
// DISASM-NEXT:   201370:       jmpq    *0x213a(%rip)
// DISASM-NEXT:   201376:       pushq   $0x0
// DISASM-NEXT:   20137b:       jmp     0x201340 <.plt>
// DISASM-NEXT:   201380:       jmpq    *0x2132(%rip)
// DISASM-NEXT:   201386:       pushq   $0x1
// DISASM-NEXT:   20138b:       jmp     0x201340 <.plt>

// Test that --shuffle-sections does not affect the order of relocations and that
// we still place IRELATIVE relocations last. Check both random seed (0) and an
// arbitrary seed that was known to break the order of relocations previously (3).
// RUN: ld.lld --shuffle-sections=3 %t.so %t.o -o %tout2
// RUN: llvm-readobj --relocations %tout2 | FileCheck %s --check-prefix=SHUFFLE
// RUN: ld.lld --shuffle-sections=0 %t.so %t.o -o %tout3
// RUN: llvm-readobj --relocations %tout3 | FileCheck %s --check-prefix=SHUFFLE

// SHUFFLE:      Section {{.*}} .rela.dyn {
// SHUFFLE-NEXT:   R_X86_64_GLOB_DAT
// SHUFFLE-NEXT:   R_X86_64_IRELATIVE
// SHUFFLE-NEXT:   R_X86_64_IRELATIVE
// SHUFFLE-NEXT: }

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
 call bar2
 call zed2
 jmp *bar3@GOTPCREL(%rip)
