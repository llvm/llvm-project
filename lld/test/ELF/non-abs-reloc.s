// REQUIRES: x86
// RUN: split-file %s %t
// RUN: llvm-mc -filetype=obj -triple=x86_64 %t/asm -o %t.o
// RUN: ld.lld -T %t/lds %t.o -o %t.exe 2>&1 | FileCheck %s
// CHECK:      warning: {{.*}}.o:(.nonalloc1+0x1): has non-ABS relocation R_X86_64_PC32 against symbol '_start'
// CHECK-NEXT: warning: {{.*}}.o:(.nonalloc1+0x6): has non-ABS relocation R_X86_64_PC32 against symbol '_start'

// RUN: llvm-objdump -D --no-show-raw-insn %t.exe | FileCheck --check-prefix=DISASM %s
// DISASM:      Disassembly of section .nonalloc:
// DISASM-EMPTY:
// DISASM-NEXT: <.nonalloc>:
// DISASM-NEXT:   0: nop
// DISASM-NEXT:   1: callq 0x0
// DISASM-NEXT:   6: callq 0x0

//--- lds
SECTIONS {
  .nonalloc 0 : { *(.nonalloc*) }
}
//--- asm
.globl _start
_start:
  nop

.section .nonalloc0
  nop

.section .nonalloc1
  .byte 0xe8
  .long _start - . - 4
  .byte 0xe8
  .long _start - . - 4
