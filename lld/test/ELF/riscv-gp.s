# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s -o %t.32.o
# RUN: ld.lld -pie %t.32.o -o %t.32
# RUN: llvm-readelf -S -s %t.32 | FileCheck --check-prefix=SEC32 %s
# RUN: not ld.lld -shared %t.32.o -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.64.o
# RUN: ld.lld -pie %t.64.o -o %t.64
# RUN: llvm-readelf -S -s %t.64 | FileCheck --check-prefix=SEC64 %s
# RUN: not ld.lld -shared %t.64.o -o /dev/null 2>&1 | FileCheck --check-prefix=ERR %s

## __global_pointer$ = .sdata+0x800 = 0x39c0
# SEC32: [ [[#SDATA:]]] .sdata PROGBITS {{0*}}000031c0
# SEC32: {{0*}}000039c0 0 NOTYPE GLOBAL DEFAULT [[#SDATA]] __global_pointer$

# SEC64: [ [[#SDATA:]]] .sdata PROGBITS {{0*}}000032e0
# SEC64: {{0*}}00003ae0 0 NOTYPE GLOBAL DEFAULT [[#SDATA]] __global_pointer$

# ERR: error: relocation R_RISCV_PCREL_HI20 cannot be used against symbol '__global_pointer$'; recompile with -fPIC

## -r mode does not define __global_pointer$.
# RUN: ld.lld -r %t.64.o -o %t.64.ro
# RUN: llvm-readelf -s %t.64.ro | FileCheck --check-prefix=RELOCATABLE %s

# RELOCATABLE: 0000000000000000 0 NOTYPE GLOBAL DEFAULT UND __global_pointer$

lla gp, __global_pointer$

.section .sdata,"aw"
