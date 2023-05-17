# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
# RUN: ld.lld --hash-style=sysv %t.o -o %t.so -shared
# RUN: llvm-readelf -S %t.so | FileCheck %s
# RUN: llvm-objdump --no-print-imm-hex -d %t.so | FileCheck --check-prefix=DISASM %s

# CHECK: .got.plt          PROGBITS        00003190

## 0x3190 - 0x1158 = 8248
# DISASM: 1158: {{.*}} movl $8248, %eax

.section .foo,"ax",@progbits
foo:
 movl $bar@got-., %eax # R_386_GOTPC

.local bar
bar:
