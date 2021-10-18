# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t

## Check recompile with -fPIC error message
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/shared.s -o %t/shared.o
# RUN: not ld.lld -shared %t/shared.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: relocation R_X86_64_PC32 cannot be used against symbol _shared; recompile with -fPIC
# CHECK: >>> defined in {{.*}}
# CHECK: >>> referenced by {{.*}}:(.data+0x1)

## Check patching of negative addends

# RUN: llvm-mc -filetype=obj -triple=x86_64 -defsym ADDEND=1 %t/addend.s -o %t/1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 -defsym ADDEND=2147483648 %t/addend.s -o %t/2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 -defsym ADDEND=2147483649 %t/addend.s -o %t/3.o

# RUN: ld.lld --section-start=.text=0x0 %t/1.o -o %t/1out
# RUN: ld.lld --section-start=.text=0x0 %t/2.o -o %t/2out
# RUN: not ld.lld --section-start=.text=0x0 %t/3.o -o /dev/null 2>&1 | FileCheck %s --check-prefix RANGE

# RANGE: relocation R_X86_64_PC32 out of range

# RUN: llvm-readelf --hex-dump=.text %t/1out | FileCheck %s -DADDEND=ffffffff --check-prefix DUMP
# RUN: llvm-readelf --hex-dump=.text %t/2out | FileCheck %s -DADDEND=00000080 --check-prefix DUMP

# DUMP:  0x00000000 [[ADDEND]]

#--- shared.s
.data
 .byte 0xe8
 .long _shared - .

#--- addend.s
.text
.globl _start
_start:
    .reloc ., R_X86_64_PC32, .text-ADDEND
    .space 4
