// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
// RUN: ld.lld -pie %t.o -o %t2
// RUN: llvm-objdump -d --no-show-raw-insn %t2 | FileCheck --check-prefix=PIE %s

.globl _start
_start:
  call __preinit_array_start
  call __preinit_array_end
  call __init_array_start
  call __init_array_end
  call __fini_array_start
  call __fini_array_end

// With no .init_array section the symbols resolve to .text.
// 0x201120 - (0x201120 + 5) = -5
// 0x201120 - (0x201125 + 5) = -10
// ...

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT:  _start:
// CHECK-NEXT:   201120:       callq    -5
// CHECK-NEXT:                 callq    -10
// CHECK-NEXT:                 callq    -15
// CHECK-NEXT:                 callq    -20
// CHECK-NEXT:                 callq    -25
// CHECK-NEXT:                 callq    -30

// In position-independent binaries, they resolve to .text too.

// PIE:      Disassembly of section .text:
// PIE-EMPTY:
// PIE-NEXT: _start:
// PIE-NEXT:     1210:       callq   -5
// PIE-NEXT:                 callq   -10
// PIE-NEXT:                 callq   -15
// PIE-NEXT:                 callq   -20
// PIE-NEXT:                 callq   -25
// PIE-NEXT:                 callq   -30
