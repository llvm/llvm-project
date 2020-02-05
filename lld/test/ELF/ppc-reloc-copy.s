# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc %p/Inputs/relocation-copy.s -o %t1.32.o
# RUN: ld.lld -shared %t1.32.o -soname=so -o %t1.32.so
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.32.o
# RUN: ld.lld %t.32.o %t1.32.so -o %t.32
# RUN: llvm-readobj -r %t.32 | FileCheck --check-prefix=REL32 %s
# RUN: llvm-nm -S %t.32 | FileCheck --check-prefix=NM32 %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %p/Inputs/relocation-copy.s -o %t1.64.o
# RUN: ld.lld -shared %t1.64.o -soname=so -o %t1.64.so
# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.64.o
# RUN: ld.lld %t.64.o %t1.64.so -o %t.64
# RUN: llvm-readobj -r %t.64 | FileCheck --check-prefix=REL64 %s
# RUN: llvm-nm -S %t.64 | FileCheck --check-prefix=NM64 %s

# REL32:      .rela.dyn {
# REL32-NEXT:   0x10030210 R_PPC_COPY x 0x0
# REL32-NEXT: }

# NM32: 10030210 00000004 B x

# REL64:      .rela.dyn {
# REL64-NEXT:   0x10030350 R_PPC64_COPY x 0x0
# REL64-NEXT: }

# NM64: 0000000010030350 0000000000000004 B x

lis 3, x@ha
lwz 3, x@l(3)
