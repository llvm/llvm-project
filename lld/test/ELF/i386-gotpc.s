// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: ld.lld --hash-style=sysv %t.o -o %t.so -shared
// RUN: llvm-readelf -S %t.so | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck --check-prefix=DISASM %s

movl $_GLOBAL_OFFSET_TABLE_, %eax

// CHECK: .got.plt          PROGBITS        00003190

// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <.text>:
// DISASM-NEXT:    1158:       movl    $8248, %eax
//                                     ^-- 0x3190 (.got.plt) - 0x1158 = 8248
