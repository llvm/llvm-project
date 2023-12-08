# REQUIRES: x86

## Test canonical PLT can be created for -no-pie and -pie.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %p/Inputs/canonical-plt-pcrel.s -o %t1.o
# RUN: ld.lld %t1.o -o %t1.so -shared -soname=so

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# RUN: ld.lld %t.o %t1.so -o %t -pie
# RUN: llvm-readobj -r %t | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:   .rela.plt {
# CHECK-NEXT:     R_X86_64_JUMP_SLOT func 0x0
# CHECK-NEXT:     R_X86_64_JUMP_SLOT ifunc 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.rodata
  .quad func    # STT_FUNC
  .quad ifunc   # STT_GNU_IFUNC
