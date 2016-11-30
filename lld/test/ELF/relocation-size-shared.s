// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/relocation-size-shared.s -o %tso.o
// RUN: ld.lld -shared %tso.o -o %tso
// RUN: ld.lld %t.o %tso -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=RELOCSHARED %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

// RELOCSHARED:       Relocations [
// RELOCSHARED-NEXT:  Section ({{.*}}) .rela.dyn {
// RELOCSHARED-NEXT:    0x203018 R_X86_64_SIZE64 fooshared 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    0x203020 R_X86_64_SIZE64 fooshared 0x0
// RELOCSHARED-NEXT:    0x203028 R_X86_64_SIZE64 fooshared 0x1
// RELOCSHARED-NEXT:    0x203048 R_X86_64_SIZE32 fooshared 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    0x20304F R_X86_64_SIZE32 fooshared 0x0
// RELOCSHARED-NEXT:    0x203056 R_X86_64_SIZE32 fooshared 0x1
// RELOCSHARED-NEXT:  }
// RELOCSHARED-NEXT:]

// DISASM:      Disassembly of section test
// DISASM:      _data:
// DISASM-NEXT: 203000: 19 00
// DISASM-NEXT: 203002: 00 00
// DISASM-NEXT: 203004: 00 00
// DISASM-NEXT: 203006: 00 00
// DISASM-NEXT: 203008: 1a 00
// DISASM-NEXT: 20300a: 00 00
// DISASM-NEXT: 20300c: 00 00
// DISASM-NEXT: 20300e: 00 00
// DISASM-NEXT: 203010: 1b 00
// DISASM-NEXT: 203012: 00 00
// DISASM-NEXT: 203014: 00 00
// DISASM-NEXT: 203016: 00 00
// DISASM-NEXT: 203018: 00 00
// DISASM-NEXT: 20301a: 00 00
// DISASM-NEXT: 20301c: 00 00
// DISASM-NEXT: 20301e: 00 00
// DISASM-NEXT: 203020: 00 00
// DISASM-NEXT: 203022: 00 00
// DISASM-NEXT: 203024: 00 00
// DISASM-NEXT: 203026: 00 00
// DISASM-NEXT: 203028: 00 00
// DISASM-NEXT: 20302a: 00 00
// DISASM-NEXT: 20302c: 00 00
// DISASM-NEXT: 20302e: 00 00
// DISASM:      _start:
// DISASM-NEXT: 203030: 8b 04 25 19 00 00 00 movl 25, %eax
// DISASM-NEXT: 203037: 8b 04 25 1a 00 00 00 movl 26, %eax
// DISASM-NEXT: 20303e: 8b 04 25 1b 00 00 00 movl 27, %eax
// DISASM-NEXT: 203045: 8b 04 25 00 00 00 00 movl 0, %eax
// DISASM-NEXT: 20304c: 8b 04 25 00 00 00 00 movl 0, %eax
// DISASM-NEXT: 203053: 8b 04 25 00 00 00 00 movl 0, %eax

.data
.global foo
.type foo,%object
.size foo,26
foo:
.zero 26

.section test, "awx"
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
