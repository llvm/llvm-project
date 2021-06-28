// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/wrap-no-real.s -o %t2.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/wrap-no-real2.s -o %t3.o
// RUN: ld.lld -o %t3.so -shared --soname=t3 %t3.o

// RUN: ld.lld -o %t %t1.o %t2.o -wrap foo
// RUN: llvm-objdump -d %t | FileCheck %s
// RUN: llvm-readelf -s -x .got %t | FileCheck --check-prefix=READELF --implicit-check-not=__real_ %s

// CHECK: <_start>:
// CHECK-NEXT: movq {{.*}}(%rip), %rax  # 0x2021a8
// CHECK-NEXT: movq {{.*}}(%rip), %rbx  # 0x2021a8
// CHECK-NEXT: movq {{.*}}(%rip), %rcx  # 0x2021b0

// READELF:      0000000000011000  0 NOTYPE GLOBAL DEFAULT ABS foo
// READELF:      0000000000011010  0 NOTYPE GLOBAL DEFAULT ABS __wrap_foo
// READELF:      Hex dump of section '.got':
// READELF-NEXT: 0x[[#%x,ADDR:]] 10100100 00000000 00100100 00000000

// RUN: ld.lld -o %t2 %t1.o %t2.o %t3.so --wrap foo
// RUN: llvm-objdump -d %t2 | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-readelf -s -x .got %t2 | FileCheck --check-prefix=READELF --implicit-check-not=__real_ %s

// CHECK2: <_start>:
// CHECK2-NEXT: movq {{.*}}(%rip), %rax  # 0x2022e0
// CHECK2-NEXT: movq {{.*}}(%rip), %rbx  # 0x2022e0
// CHECK2-NEXT: movq {{.*}}(%rip), %rcx  # 0x2022e8

.global _start
_start:
  mov foo@gotpcrel(%rip), %rax
  mov __wrap_foo@gotpcrel(%rip), %rbx
  mov __real_foo@gotpcrel(%rip), %rcx
