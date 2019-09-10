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
// CHECK-NEXT:     0x203028 R_X86_64_IRELATIVE - 0x201000
// CHECK-NEXT:     0x203030 R_X86_64_IRELATIVE - 0x201001
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rela.plt {
// CHECK-NEXT:     0x203018 R_X86_64_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x203020 R_X86_64_JUMP_SLOT zed2 0x0
// CHECK-NEXT:   }

// Check that .got.plt entries point back to PLT header
// GOTPLT: Contents of section .got.plt:
// GOTPLT-NEXT:  203000 00202000 00000000 00000000 00000000
// GOTPLT-NEXT:  203010 00000000 00000000 36102000 00000000
// GOTPLT-NEXT:  203020 46102000 00000000 56102000 00000000
// GOTPLT-NEXT:  203030 66102000 00000000

// Check that the PLTRELSZ tag does not include the IRELATIVE relocations
// CHECK: DynamicSection [
// CHECK:   0x0000000000000008 RELASZ               48 (bytes)
// CHECK:   0x0000000000000002 PLTRELSZ             48 (bytes)

// Check that a PLT header is written and the ifunc entries appear last
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: foo:
// DISASM-NEXT:   201000:       retq
// DISASM:      bar:
// DISASM-NEXT:   201001:       retq
// DISASM:      _start:
// DISASM-NEXT:   201002:       callq   73
// DISASM-NEXT:   201007:       callq   84
// DISASM-NEXT:                 callq   {{.*}} <bar2@plt>
// DISASM-NEXT:                 callq   {{.*}} <zed2@plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: .plt:
// DISASM-NEXT:   201020:       pushq   8162(%rip)
// DISASM-NEXT:   201026:       jmpq    *8164(%rip)
// DISASM-NEXT:   20102c:       nopl    (%rax)
// DISASM-EMPTY:
// DISASM-NEXT:   bar2@plt:
// DISASM-NEXT:   201030:       jmpq    *8162(%rip)
// DISASM-NEXT:   201036:       pushq   $0
// DISASM-NEXT:   20103b:       jmp     -32 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT:   zed2@plt:
// DISASM-NEXT:   201040:       jmpq    *8154(%rip)
// DISASM-NEXT:   201046:       pushq   $1
// DISASM-NEXT:   20104b:       jmp     -48 <.plt>
// DISASM-NEXT:   201050:       jmpq    *8146(%rip)
// DISASM-NEXT:   201056:       pushq   $0
// DISASM-NEXT:   20105b:       jmp     -32 <zed2@plt>
// DISASM-NEXT:   201060:       jmpq    *8138(%rip)
// DISASM-NEXT:   201066:       pushq   $1
// DISASM-NEXT:   20106b:       jmp     -48 <zed2@plt>

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
