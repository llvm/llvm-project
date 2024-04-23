# REQUIRES: systemz
# RUN: rm -rf %t && split-file %s %t

## Check recompile with -fPIC error message
# RUN: llvm-mc -filetype=obj -triple=s390x-unknown-linux %t/shared.s -o %t/shared.o
# RUN: not ld.lld -shared %t/shared.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: relocation R_390_PC16 cannot be used against symbol '_shared'; recompile with -fPIC
# CHECK: >>> defined in {{.*}}
# CHECK: >>> referenced by {{.*}}:(.data+0x1)

## Check patching of negative addends

# RUN: llvm-mc -filetype=obj -triple=s390x -defsym ADDEND=1 %t/addend.s -o %t/1.o
# RUN: llvm-mc -filetype=obj -triple=s390x -defsym ADDEND=32768 %t/addend.s -o %t/2.o
# RUN: llvm-mc -filetype=obj -triple=s390x -defsym ADDEND=32769 %t/addend.s -o %t/3.o

# RUN: ld.lld --section-start=.text=0x0 %t/1.o -o %t/1out
# RUN: ld.lld --section-start=.text=0x0 %t/2.o -o %t/2out
# RUN: not ld.lld --section-start=.text=0x0 %t/3.o -o /dev/null 2>&1 | FileCheck %s -DFILE=%t/3.o --check-prefix RANGE

# RANGE: error: [[FILE]]:(.text+0x0): relocation R_390_PC16 out of range

# RUN: llvm-readelf --hex-dump=.text %t/1out | FileCheck %s -DADDEND=ffff --check-prefix DUMP
# RUN: llvm-readelf --hex-dump=.text %t/2out | FileCheck %s -DADDEND=8000 --check-prefix DUMP

# DUMP:  0x00000000 [[ADDEND]]

#--- shared.s
.data
 .byte 0xe8
 .word _shared - .

#--- addend.s
.text
.globl _start
_start:
    .reloc ., R_390_PC16, .text-ADDEND
    .space 2
