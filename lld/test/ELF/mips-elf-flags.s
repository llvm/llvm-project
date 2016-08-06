# Check generation of MIPS specific ELF header flags.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %S/Inputs/mips-dynamic.s -o %t-so.o
# RUN: ld.lld %t-so.o -shared -o %t.so
# RUN: llvm-readobj -h %t.so | FileCheck -check-prefix=SO %s
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-readobj -h %t.exe | FileCheck -check-prefix=EXE %s
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r2 %s -o %t-r2.o
# RUN: ld.lld %t-r2.o -o %t-r2.exe
# RUN: llvm-readobj -h %t-r2.exe | FileCheck -check-prefix=EXE-R2 %s
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r6 %s -o %t-r6.o
# RUN: ld.lld %t-r6.o -o %t-r6.exe
# RUN: llvm-readobj -h %t-r6.exe | FileCheck -check-prefix=EXE-R6 %s

# REQUIRES: mips

  .text
  .globl  __start
__start:
  nop

# SO:      Flags [
# SO-NEXT:   EF_MIPS_ABI_O32
# SO-NEXT:   EF_MIPS_ARCH_32
# SO-NEXT:   EF_MIPS_CPIC
# SO-NEXT:   EF_MIPS_PIC
# SO-NEXT: ]

# EXE:      Flags [
# EXE-NEXT:   EF_MIPS_ABI_O32
# EXE-NEXT:   EF_MIPS_ARCH_32
# EXE-NEXT:   EF_MIPS_CPIC
# EXE-NEXT: ]

# EXE-R2:      Flags [
# EXE-R2-NEXT:   EF_MIPS_ABI_O32
# EXE-R2-NEXT:   EF_MIPS_ARCH_32R2
# EXE-R2-NEXT:   EF_MIPS_CPIC
# EXE-R2-NEXT: ]

# EXE-R6:      Flags [
# EXE-R6-NEXT:   EF_MIPS_ABI_O32
# EXE-R6-NEXT:   EF_MIPS_ARCH_32R6
# EXE-R6-NEXT:   EF_MIPS_CPIC
# EXE-R6-NEXT:   EF_MIPS_NAN2008
# EXE-R6-NEXT: ]
