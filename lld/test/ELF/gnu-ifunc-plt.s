// REQUIRES: x86

/// For non-preemptable ifunc, place ifunc PLT entries after regular PLT entries.

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %S/Inputs/shared2-x86-64.s -o %t1.o
// RUN: ld.lld %t1.o --shared -soname=so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld --hash-style=sysv %t.so %t.o -o %tout
// RUN: llvm-objdump -d --no-show-raw-insn %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -s %tout | FileCheck %s --check-prefix=GOTPLT
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

// Check that the IRELATIVE relocations are after the JUMP_SLOT in the plt
// CHECK: Relocations [
// CHECK-NEXT:   Section (4) .rela.dyn {
// CHECK-NEXT:     0x203458 R_X86_64_IRELATIVE - 0x2012D8
// CHECK-NEXT:     0x203460 R_X86_64_IRELATIVE - 0x2012D9
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rela.plt {
// CHECK-NEXT:     0x203448 R_X86_64_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x203450 R_X86_64_JUMP_SLOT zed2 0x0
// CHECK-NEXT:   }

// Check that .got.plt entries point back to PLT header
// GOTPLT: Contents of section .got.plt:
// GOTPLT-NEXT:  203430 40232000 00000000 00000000 00000000
// GOTPLT-NEXT:  203440 00000000 00000000 06132000 00000000
// GOTPLT-NEXT:  203450 16132000 00000000 26132000 00000000
// GOTPLT-NEXT:  203460 36132000 00000000

// Check that the PLTRELSZ tag does not include the IRELATIVE relocations
// CHECK: DynamicSection [
// CHECK:   0x0000000000000008 RELASZ               48 (bytes)
// CHECK:   0x0000000000000002 PLTRELSZ             48 (bytes)

// Check that a PLT header is written and the ifunc entries appear last
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: foo:
// DISASM-NEXT:   2012d8:       retq
// DISASM:      bar:
// DISASM-NEXT:   2012d9:       retq
// DISASM:      _start:
// DISASM-NEXT:   2012da:       callq   65
// DISASM-NEXT:   2012df:       callq   76
// DISASM-NEXT:                 callq   {{.*}} <bar2@plt>
// DISASM-NEXT:                 callq   {{.*}} <zed2@plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: .plt:
// DISASM-NEXT:   2012f0:       pushq   8514(%rip)
// DISASM-NEXT:   2012f6:       jmpq    *8516(%rip)
// DISASM-NEXT:   2012fc:       nopl    (%rax)
// DISASM-EMPTY:
// DISASM-NEXT:   bar2@plt:
// DISASM-NEXT:   201300:       jmpq    *8514(%rip)
// DISASM-NEXT:   201306:       pushq   $0
// DISASM-NEXT:   20130b:       jmp     -32 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT:   zed2@plt:
// DISASM-NEXT:   201310:       jmpq    *8506(%rip)
// DISASM-NEXT:   201316:       pushq   $1
// DISASM-NEXT:   20131b:       jmp     -48 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .iplt:
// DISASM-EMPTY:
// DISASM-NEXT: .iplt:
// DISASM-NEXT:   201320:       jmpq    *8498(%rip)
// DISASM-NEXT:   201326:       pushq   $0
// DISASM-NEXT:   20132b:       jmp     -64 <.plt>
// DISASM-NEXT:   201330:       jmpq    *8490(%rip)
// DISASM-NEXT:   201336:       pushq   $1
// DISASM-NEXT:   20133b:       jmp     -80 <.plt>

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
