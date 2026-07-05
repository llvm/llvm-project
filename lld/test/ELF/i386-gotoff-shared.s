// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t.so -shared
// RUN: llvm-readelf -S %t.so | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck --check-prefix=DISASM %s

bar:
        movl    bar@GOTOFF(%ebx), %eax
        mov     bar@GOT, %eax

// CHECK: .got.plt          PROGBITS        000031e0

// 0x1178 - 0x31e0 (.got.plt) = -8296

// DISASM:  1178:       movl    -8296(%ebx), %eax
