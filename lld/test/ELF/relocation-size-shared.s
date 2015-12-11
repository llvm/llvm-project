// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/relocation-size-shared.s -o %tso.o
// RUN: ld.lld -shared %tso.o -o %tso
// RUN: ld.lld %t.o %tso -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=RELOCSHARED %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

// RELOCSHARED:       Relocations [
// RELOCSHARED-NEXT:  Section ({{.*}}) .rela.dyn {
// RELOCSHARED-NEXT:    0x11018 R_X86_64_SIZE64 fooshared 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    0x11020 R_X86_64_SIZE64 fooshared 0x0
// RELOCSHARED-NEXT:    0x11028 R_X86_64_SIZE64 fooshared 0x1
// RELOCSHARED-NEXT:    0x11048 R_X86_64_SIZE32 fooshared 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    0x1104F R_X86_64_SIZE32 fooshared 0x0
// RELOCSHARED-NEXT:    0x11056 R_X86_64_SIZE32 fooshared 0x1
// RELOCSHARED-NEXT:  }
// RELOCSHARED-NEXT:]

// DISASM:      Disassembly of section .text:
// DISASM:      _data:
// DISASM-NEXT: 11000: 19 00
// DISASM-NEXT: 11002: 00 00
// DISASM-NEXT: 11004: 00 00
// DISASM-NEXT: 11006: 00 00
// DISASM-NEXT: 11008: 1a 00
// DISASM-NEXT: 1100a: 00 00
// DISASM-NEXT: 1100c: 00 00
// DISASM-NEXT: 1100e: 00 00
// DISASM-NEXT: 11010: 1b 00
// DISASM-NEXT: 11012: 00 00
// DISASM-NEXT: 11014: 00 00
// DISASM-NEXT: 11016: 00 00
// DISASM-NEXT: 11018: 00 00
// DISASM-NEXT: 1101a: 00 00
// DISASM-NEXT: 1101c: 00 00
// DISASM-NEXT: 1101e: 00 00
// DISASM-NEXT: 11020: 00 00
// DISASM-NEXT: 11022: 00 00
// DISASM-NEXT: 11024: 00 00
// DISASM-NEXT: 11026: 00 00
// DISASM-NEXT: 11028: 00 00
// DISASM-NEXT: 1102a: 00 00
// DISASM-NEXT: 1102c: 00 00
// DISASM-NEXT: 1102e: 00 00
// DISASM:      _start:
// DISASM-NEXT: 11030: 8b 04 25 19 00 00 00 movl 25, %eax
// DISASM-NEXT: 11037: 8b 04 25 1a 00 00 00 movl 26, %eax
// DISASM-NEXT: 1103e: 8b 04 25 1b 00 00 00 movl 27, %eax
// DISASM-NEXT: 11045: 8b 04 25 00 00 00 00 movl 0, %eax
// DISASM-NEXT: 1104c: 8b 04 25 00 00 00 00 movl 0, %eax
// DISASM-NEXT: 11053: 8b 04 25 00 00 00 00 movl 0, %eax

.data
.global foo
.type foo,%object
.size foo,26
foo:
.zero 26

.text
_data:
  // R_X86_64_SIZE64:
  .quad foo@SIZE-1
  .quad foo@SIZE
  .quad foo@SIZE+1
  .quad fooshared@SIZE-1
  .quad fooshared@SIZE
  .quad fooshared@SIZE+1

.globl _start
_start:
  // R_X86_64_SIZE32:
  movl foo@SIZE-1,%eax
  movl foo@SIZE,%eax
  movl foo@SIZE+1,%eax
  movl fooshared@SIZE-1,%eax
  movl fooshared@SIZE,%eax
  movl fooshared@SIZE+1,%eax
