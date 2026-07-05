# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 %p/Inputs/relocation-copy.s -o %t1.o
# RUN: ld.lld -shared %t1.o -soname=t1.so -o %t1.so
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.o
# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefixes=REL,REL32 %s
# RUN: llvm-nm -S %t | FileCheck --check-prefix=NM32 %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 %p/Inputs/relocation-copy.s -o %t1.o
# RUN: ld.lld -shared %t1.o -soname=t1.so -o %t1.so
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefixes=REL,REL64 %s
# RUN: llvm-nm -S %t | FileCheck --check-prefix=NM64 %s

# REL:        .rela.dyn {
# REL32-NEXT:   0x13210 R_RISCV_COPY x 0x0
# REL64-NEXT:   0x13360 R_RISCV_COPY x 0x0
# REL-NEXT:   }

# NM32: 00013210 00000004 B x
# NM64: 0000000000013360 0000000000000004 B x

la a0, x
