# REQUIRES: x86

## Test copy relocations can be created for -no-pie and -pie.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/copy-rel.s -o %t1.o
# RUN: ld.lld %t1.o -o %t1.so -shared -soname=so

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# RUN: ld.lld %t.o %t1.so -o %t -pie
# RUN: llvm-readobj -r %t | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:   .rela.dyn {
# CHECK-NEXT:     R_X86_64_COPY bar 0x0
# CHECK-NEXT:     R_X86_64_COPY foo 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.global _start
_start:
  mov $foo - ., %eax
  movabs $bar, %rax
