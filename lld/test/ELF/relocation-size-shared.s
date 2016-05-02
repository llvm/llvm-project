// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/relocation-size-shared.s -o %tso.o
// RUN: ld.lld -shared %tso.o -o %tso
// RUN: ld.lld %t.o %tso -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=RELOCSHARED %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s

// RELOCSHARED:       Relocations [
// RELOCSHARED-NEXT:  Section ({{.*}}) .rela.dyn {
// RELOCSHARED-NEXT:    0x13018 R_X86_64_SIZE64 fooshared 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    0x13020 R_X86_64_SIZE64 fooshared 0x0
// RELOCSHARED-NEXT:    0x13028 R_X86_64_SIZE64 fooshared 0x1
// RELOCSHARED-NEXT:    0x13048 R_X86_64_SIZE32 fooshared 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    0x1304F R_X86_64_SIZE32 fooshared 0x0
// RELOCSHARED-NEXT:    0x13056 R_X86_64_SIZE32 fooshared 0x1
// RELOCSHARED-NEXT:  }
// RELOCSHARED-NEXT:]

// DISASM:      Disassembly of section test
// DISASM:      _data:
// DISASM-NEXT: 13000: 19 00
// DISASM-NEXT: 13002: 00 00
// DISASM-NEXT: 13004: 00 00
// DISASM-NEXT: 13006: 00 00
// DISASM-NEXT: 13008: 1a 00
// DISASM-NEXT: 1300a: 00 00
// DISASM-NEXT: 1300c: 00 00
// DISASM-NEXT: 1300e: 00 00
// DISASM-NEXT: 13010: 1b 00
// DISASM-NEXT: 13012: 00 00
// DISASM-NEXT: 13014: 00 00
// DISASM-NEXT: 13016: 00 00
// DISASM-NEXT: 13018: 00 00
// DISASM-NEXT: 1301a: 00 00
// DISASM-NEXT: 1301c: 00 00
// DISASM-NEXT: 1301e: 00 00
// DISASM-NEXT: 13020: 00 00
// DISASM-NEXT: 13022: 00 00
// DISASM-NEXT: 13024: 00 00
// DISASM-NEXT: 13026: 00 00
// DISASM-NEXT: 13028: 00 00
// DISASM-NEXT: 1302a: 00 00
// DISASM-NEXT: 1302c: 00 00
// DISASM-NEXT: 1302e: 00 00
// DISASM:      _start:
// DISASM-NEXT: 13030: 8b 04 25 19 00 00 00 movl 25, %eax
// DISASM-NEXT: 13037: 8b 04 25 1a 00 00 00 movl 26, %eax
// DISASM-NEXT: 1303e: 8b 04 25 1b 00 00 00 movl 27, %eax
// DISASM-NEXT: 13045: 8b 04 25 00 00 00 00 movl 0, %eax
// DISASM-NEXT: 1304c: 8b 04 25 00 00 00 00 movl 0, %eax
// DISASM-NEXT: 13053: 8b 04 25 00 00 00 00 movl 0, %eax

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
