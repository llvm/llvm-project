// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t.so
// RUN: llvm-readelf -S %t.so | FileCheck --check-prefix=SEC %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s

// SEC:      .got PROGBITS 000021f0
// SEC:      .got.plt PROGBITS 000031f4

// CHECK: <foo>:
// CHECK-NEXT: 1194: movl    -4100(%ebx), %eax
// CHECK-NEXT:       movl    -4092(%ebx), %eax

foo:
        movl bar@GOT(%ebx), %eax
        movl bar+8@GOT(%ebx), %eax
