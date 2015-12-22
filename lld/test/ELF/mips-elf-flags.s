# Check generation of MIPS specific ELF header flags.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -shared -o %t.so
# RUN: llvm-readobj -h %t.so | FileCheck -check-prefix=SO %s
# RUN: ld.lld %t.o -o %t.exe
# RUN: llvm-readobj -h %t.exe | FileCheck -check-prefix=EXE %s

# REQUIRES: mips

  .text
  .globl  __start
__start:
  nop

# SO:      Flags [
# SO-NEXT:   EF_MIPS_ABI_O32
# SO-NEXT:   EF_MIPS_ARCH_32R2
# SO-NEXT:   EF_MIPS_CPIC
# SO-NEXT:   EF_MIPS_PIC
# SO-NEXT: ]

# EXE:      Flags [
# EXE-NEXT:   EF_MIPS_ABI_O32
# EXE-NEXT:   EF_MIPS_ARCH_32R2
# EXE-NEXT:   EF_MIPS_CPIC
# EXE-NEXT: ]
