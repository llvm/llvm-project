## Checks handling of function names passed via -funcs-no-regex

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -nostdlib
# RUN: llvm-bolt %t.exe -o %t.out -funcs-no-regex=func -print-cfg | FileCheck %s
# CHECK: Binary Function "func/1"

.globl _start
.type _start, @function
_start:
  ret
  .size _start, .-_start

.type func, @function
func:
  ud2
