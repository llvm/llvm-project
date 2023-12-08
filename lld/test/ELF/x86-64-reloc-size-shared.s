// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/relocation-size-shared.s -o %tso.o
// RUN: ld.lld -shared %tso.o -soname=so -o %t1.so
// RUN: ld.lld %t.o %t1.so -o %t
// RUN: llvm-readobj -r %t | FileCheck --check-prefix=RELOCSHARED %s
// RUN: llvm-readelf -x .data %t | FileCheck --check-prefix=DATA %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASM %s

// RELOCSHARED:       Relocations [
// RELOCSHARED-NEXT:  Section ({{.*}}) .rela.dyn {
// RELOCSHARED-NEXT:    R_X86_64_SIZE32 fooshared 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    R_X86_64_SIZE32 fooshared 0x0
// RELOCSHARED-NEXT:    R_X86_64_SIZE32 fooshared 0x1
// RELOCSHARED-NEXT:    R_X86_64_SIZE64 fooshared 0xFFFFFFFFFFFFFFFF
// RELOCSHARED-NEXT:    R_X86_64_SIZE64 fooshared 0x0
// RELOCSHARED-NEXT:    R_X86_64_SIZE64 fooshared 0x1
// RELOCSHARED-NEXT:  }
// RELOCSHARED-NEXT:]

// DATA:      section '.data':
// DATA-NEXT:   00000000 00000000 00000000 00000000
// DATA-NEXT:   00000000 00000000 00001900 00000000
// DATA-NEXT:   00001a00 00000000 00001b00 00000000
// DATA-NEXT:   00000000 00000000 00000000 00000000
// DATA-NEXT:   00000000 00000000 0000

// DISASM:      <_start>:
// DISASM-NEXT:   movl 25, %eax
// DISASM-NEXT:   movl 26, %eax
// DISASM-NEXT:   movl 27, %eax
// DISASM-NEXT:   movl 0, %eax
// DISASM-NEXT:   movl 0, %eax
// DISASM-NEXT:   movl 0, %eax

.data
.global foo
.type foo,%object
.size foo,26
foo:
.zero 26

  // R_X86_64_SIZE64:
  .quad foo@SIZE-1
  .quad foo@SIZE
  .quad foo@SIZE+1
  .quad fooshared@SIZE-1
  .quad fooshared@SIZE
  .quad fooshared@SIZE+1

.section test, "awx"
.globl _start
_start:
  // R_X86_64_SIZE32:
  movl foo@SIZE-1,%eax
  movl foo@SIZE,%eax
  movl foo@SIZE+1,%eax
  movl fooshared@SIZE-1,%eax
  movl fooshared@SIZE,%eax
  movl fooshared@SIZE+1,%eax
