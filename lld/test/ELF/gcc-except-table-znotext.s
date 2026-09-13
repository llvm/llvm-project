# REQUIRES: x86
## A symbolic relocation for -z notext in .gcc_except_table could emit a dynamic
## relocation, but we avoid that and resolve it at link time, using a copy
## relocation where one is needed. This matches GNU ld and gold, and mirrors the
## treatment of .eh_frame.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld -shared b.o -o b.so -soname b.so

# RUN: ld.lld a.o b.so -o a
# RUN: llvm-readelf -r a | FileCheck %s
# RUN: ld.lld -z notext a.o b.so -o a
# RUN: llvm-readelf -r a | FileCheck %s

## Per-function sections created by -ffunction-sections get the same treatment.
# RUN: llvm-mc -filetype=obj -triple=x86_64 c.s -o c.o
# RUN: ld.lld -z notext c.o b.so -o c
# RUN: llvm-readelf -r c | FileCheck %s

# CHECK:     R_X86_64_COPY {{.*}} obj + 0
# CHECK-NOT: R_X86_64_64

#--- a.s
.globl _start
_start:
  ret

.section .gcc_except_table,"a",@progbits
  .quad obj

#--- c.s
.globl _start
_start:
  ret

.section .gcc_except_table._start,"a",@progbits
  .quad obj

#--- b.s
.data
.globl obj
.type obj, @object
.size obj, 8
obj:
  .quad 0
