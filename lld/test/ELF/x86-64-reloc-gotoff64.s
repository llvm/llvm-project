// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld -shared -o %t.so %t.o
// RUN: llvm-readelf -sections %t.so | FileCheck %s
// RUN: llvm-objdump -d %t.so | FileCheck -check-prefix=DISASM %s

// CHECK: .dynamic DYNAMIC  0000000000002000 002000
// CHECK: .got     PROGBITS 0000000000002070 002070

// DISASM: 1000: 48 ba 90 ff ff ff ff ff ff ff   movabsq $-112, %rdx

.global _start
.weak _DYNAMIC
.hidden _DYNAMIC
_start:
  movabsq $_DYNAMIC@GOTOFF, %rdx
