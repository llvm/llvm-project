# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t-dup.o
# RUN: not %lld -o /dev/null %t-dup.o %t.o 2>&1 | FileCheck %s -DFILE_1=%t-dup.o -DFILE_2=%t.o
# RUN: not %lld -o /dev/null %t.o %t.o 2>&1 | FileCheck %s -DFILE_1=%t.o -DFILE_2=%t.o

# CHECK:      error: duplicate symbol: _main
# CHECK-NEXT: >>> defined in [[FILE_1]]
# CHECK-NEXT: >>> defined in [[FILE_2]]

.text
.global _main
_main:
  mov $0, %rax
  ret
