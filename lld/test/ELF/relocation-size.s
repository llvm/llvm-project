// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=NORELOC %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASM %s
// RUN: ld.lld -shared %t.o -o %t1
// RUN: llvm-readobj -r %t1 | FileCheck --check-prefix=RELOCSHARED %s
// RUN: llvm-objdump -d %t1 | FileCheck --check-prefix=DISASMSHARED %s

// NORELOC:      Relocations [
// NORELOC-NEXT: ]

// DISASM:      Disassembly of section test:
// DISASM-NEXT: _data:
// DISASM-NEXT: 12000: 19 00
// DISASM-NEXT: 12002: 00 00
// DISASM-NEXT: 12004: 00 00
// DISASM-NEXT: 12006: 00 00
// DISASM-NEXT: 12008: 1a 00
// DISASM-NEXT: 1200a: 00 00
// DISASM-NEXT: 1200c: 00 00
// DISASM-NEXT: 1200e: 00 00
// DISASM-NEXT: 12010: 1b 00
// DISASM-NEXT: 12012: 00 00
// DISASM-NEXT: 12014: 00 00
// DISASM-NEXT: 12016: 00 00
// DISASM-NEXT: 12018: 19 00
// DISASM-NEXT: 1201a: 00 00
// DISASM-NEXT: 1201c: 00 00
// DISASM-NEXT: 1201e: 00 00
// DISASM-NEXT: 12020: 1a 00
// DISASM-NEXT: 12022: 00 00
// DISASM-NEXT: 12024: 00 00
// DISASM-NEXT: 12026: 00 00
// DISASM-NEXT: 12028: 1b 00
// DISASM-NEXT: 1202a: 00 00
// DISASM-NEXT: 1202c: 00 00
// DISASM-NEXT: 1202e: 00 00
// DISASM:      _start:
// DISASM-NEXT: 12030: 8b 04 25 19 00 00 00 movl 25, %eax
// DISASM-NEXT: 12037: 8b 04 25 1a 00 00 00 movl 26, %eax
// DISASM-NEXT: 1203e: 8b 04 25 1b 00 00 00 movl 27, %eax
// DISASM-NEXT: 12045: 8b 04 25 19 00 00 00 movl 25, %eax
// DISASM-NEXT: 1204c: 8b 04 25 1a 00 00 00 movl 26, %eax
// DISASM-NEXT: 12053: 8b 04 25 1b 00 00 00 movl 27, %eax

// RELOCSHARED:      Relocations [
// RELOCSHARED-NEXT: Section ({{.*}}) .rela.dyn {
// RELOCSHARED-NEXT:    0x3000 R_X86_64_SIZE64 foo 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    0x3008 R_X86_64_SIZE64 foo 0x0
// RELOCSHARED-NEXT:    0x3010 R_X86_64_SIZE64 foo 0x1
// RELOCSHARED-NEXT:    0x3033 R_X86_64_SIZE32 foo 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    0x303A R_X86_64_SIZE32 foo 0x0
// RELOCSHARED-NEXT:    0x3041 R_X86_64_SIZE32 foo 0x1
// RELOCSHARED-NEXT:  }
// RELOCSHARED-NEXT: ]

// DISASMSHARED:      Disassembly of section test:
// DISASMSHARED-NEXT: _data:
// DISASMSHARED-NEXT: 3000: 00 00
// DISASMSHARED-NEXT: 3002: 00 00
// DISASMSHARED-NEXT: 3004: 00 00
// DISASMSHARED-NEXT: 3006: 00 00
// DISASMSHARED-NEXT: 3008: 00 00
// DISASMSHARED-NEXT: 300a: 00 00
// DISASMSHARED-NEXT: 300c: 00 00
// DISASMSHARED-NEXT: 300e: 00 00
// DISASMSHARED-NEXT: 3010: 00 00
// DISASMSHARED-NEXT: 3012: 00 00
// DISASMSHARED-NEXT: 3014: 00 00
// DISASMSHARED-NEXT: 3016: 00 00
// DISASMSHARED-NEXT: 3018: 19 00
// DISASMSHARED-NEXT: 301a: 00 00
// DISASMSHARED-NEXT: 301c: 00 00
// DISASMSHARED-NEXT: 301e: 00 00
// DISASMSHARED-NEXT: 3020: 1a 00
// DISASMSHARED-NEXT: 3022: 00 00
// DISASMSHARED-NEXT: 3024: 00 00
// DISASMSHARED-NEXT: 3026: 00 00
// DISASMSHARED-NEXT: 3028: 1b 00
// DISASMSHARED-NEXT: 302a: 00 00
// DISASMSHARED-NEXT: 302c: 00 00
// DISASMSHARED-NEXT: 302e: 00 00
// DISASMSHARED:      _start:
// DISASMSHARED-NEXT: 3030: 8b 04 25 00 00 00 00 movl 0, %eax
// DISASMSHARED-NEXT: 3037: 8b 04 25 00 00 00 00 movl 0, %eax
// DISASMSHARED-NEXT: 303e: 8b 04 25 00 00 00 00 movl 0, %eax
// DISASMSHARED-NEXT: 3045: 8b 04 25 19 00 00 00 movl 25, %eax
// DISASMSHARED-NEXT: 304c: 8b 04 25 1a 00 00 00 movl 26, %eax
// DISASMSHARED-NEXT: 3053: 8b 04 25 1b 00 00 00 movl 27, %eax

.data
.global foo
.type foo,%object
.size foo,26
foo:
.zero 26

.data
.global foohidden
.hidden foohidden
.type foohidden,%object
.size foohidden,26
foohidden:
.zero 26

.section test,"axw"
_data:
  // R_X86_64_SIZE64:
  .quad foo@SIZE-1
  .quad foo@SIZE
  .quad foo@SIZE+1
  .quad foohidden@SIZE-1
  .quad foohidden@SIZE
  .quad foohidden@SIZE+1
.globl _start
_start:
  // R_X86_64_SIZE32:
  movl foo@SIZE-1,%eax
  movl foo@SIZE,%eax
  movl foo@SIZE+1,%eax
  movl foohidden@SIZE-1,%eax
  movl foohidden@SIZE,%eax
  movl foohidden@SIZE+1,%eax
