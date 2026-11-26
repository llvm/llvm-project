# RUN: llvm-mc -triple=riscv64 -mattr=+c,+relax -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-readelf -r %t.exe | FileCheck %s --check-prefix=INPUT
# RUN: llvm-bolt %t.exe --reorder-functions=cdsort -o %t.bolt 2>&1 | FileCheck %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CODE

## Retained R_RISCV_ALIGN relocations describe padding already handled by
## the linker. Check that BOLT accepts them and preserves valid code.
# INPUT: R_RISCV_ALIGN
# CHECK: BOLT-INFO: enabling relocation mode
# CHECK-NOT: Failed to analyze
# CODE: <_start>:
# CODE-NEXT: jal
# CODE: ret

  .text
  .globl _start
  .type _start, @function
  .p2align 2
_start:
  call foo
  .p2align 2
  ret
  .size _start, .-_start

  .type foo, @function
foo:
  ret
  .size foo, .-foo
