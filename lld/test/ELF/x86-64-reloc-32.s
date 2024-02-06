# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/shared.s -o %t/shared.o

# CHECK: error: relocation R_X86_64_32 cannot be used against symbol '_shared'; recompile with -fPIC
# CHECK: >>> defined in {{.*}}
# CHECK: >>> referenced by {{.*}}:(.data+0x0)

## Check patching of negative addends
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/addend.s -o %t/addend.o
# RUN: not ld.lld --section-start=.text=0x0 %t/addend.o -o /dev/null 2>&1 | FileCheck %s --check-prefix RANGE

# RANGE: relocation R_X86_64_32 out of range

#--- shared.s
.data
.long _shared

#--- addend.s
.text
.globl _start
_start:
    .reloc ., R_X86_64_32, .text-1
    .space 4
