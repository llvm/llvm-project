# Check MIPS ELF ISA flag calculation if input files have different ISAs.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32 %S/Inputs/mips-dynamic.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r2 %s -o %t2.o
# RUN: ld.lld %t1.o %t2.o -o %t.exe
# RUN: llvm-readobj -h %t.exe | FileCheck -check-prefix=R1R2 %s

# Check that lld does not allow to link incompatible ISAs.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32 %S/Inputs/mips-dynamic.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mcpu=mips32r6 %s -o %t2.o
# RUN: not ld.lld %t1.o %t2.o -o %t.exe 2>&1 | FileCheck -check-prefix=R1R6 %s

# Check that lld does not allow to link incompatible ABIs.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -target-abi n32 %S/Inputs/mips-dynamic.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -target-abi o32 %s -o %t2.o
# RUN: not ld.lld %t1.o %t2.o -o %t.exe 2>&1 | FileCheck -check-prefix=N32O32 %s

# Check that lld does not allow to link modules with incompatible NAN flags.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=+nan2008 %S/Inputs/mips-dynamic.s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %s -o %t2.o
# RUN: not ld.lld %t1.o %t2.o -o %t.exe 2>&1 | FileCheck -check-prefix=NAN %s

# REQUIRES: mips

  .option pic0
  .text
  .global  __start
__start:
  nop

# R1R2:      Flags [
# R1R2-NEXT:   EF_MIPS_ABI_O32
# R1R2-NEXT:   EF_MIPS_ARCH_32R2
# R1R2-NEXT:   EF_MIPS_CPIC
# R1R2-NEXT: ]

# R1R6: target ISA 'mips32' is incompatible with 'mips32r6': {{.*}}mips-elf-flags-err.s.tmp2.o

# N32O32: target ABI 'n32' is incompatible with 'o32': {{.*}}mips-elf-flags-err.s.tmp2.o

# NAN: target -mnan=2008 is incompatible with -mnan=legacy: {{.*}}mips-elf-flags-err.s.tmp2.o
