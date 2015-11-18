// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s
// REQUIRES: x86

.globl _start
_start:
  call __preinit_array_start
  call __preinit_array_end
  call __init_array_start
  call __init_array_end
  call __fini_array_start
  call __fini_array_end

// With no .init_array section the symbols resolve to 0
// 0 - (0x11000 + 5) = -69637
// 0 - (0x11005 + 5) = -69642
// 0 - (0x1100a + 5) = -69647
// 0 - (0x1100f + 5) = -69652
// 0 - (0x11014 + 5) = -69657
// 0 - (0x11019 + 5) = -69662

// CHECK: Disassembly of section .text:
// CHECK-NEXT:  _start:
// CHECK-NEXT:   11000:    e8 fb ef fe ff     callq    -69637
// CHECK-NEXT:   11005:    e8 f6 ef fe ff     callq    -69642
// CHECK-NEXT:   1100a:    e8 f1 ef fe ff     callq    -69647
// CHECK-NEXT:   1100f:    e8 ec ef fe ff     callq    -69652
// CHECK-NEXT:   11014:    e8 e7 ef fe ff     callq    -69657
// CHECK-NEXT:   11019:    e8 e2 ef fe ff     callq    -69662
