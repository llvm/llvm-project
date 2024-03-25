# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x -defsym DISP=74565 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=s390x -defsym DISP=524287 %s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=s390x -defsym DISP=524288 %s -o %t3.o

# RUN: ld.lld --section-start=.text=0x0 %t1.o -o %t1out
# RUN: ld.lld --section-start=.text=0x0 %t2.o -o %t2out
# RUN: not ld.lld --section-start=.text=0x0 %t3.o -o /dev/null 2>&1 | FileCheck %s --check-prefix RANGE

# RANGE: relocation R_390_20 out of range: 524288 is not in [-524288, 524287]

# RUN: llvm-readelf --hex-dump=.text %t1out | FileCheck %s -DINSN="e3678345 1204" --check-prefix DUMP
# RUN: llvm-readelf --hex-dump=.text %t2out | FileCheck %s -DINSN="e3678fff 7f04" --check-prefix DUMP

# DUMP:  0x00000000 [[INSN]]

.text
.globl _start
_start:
    .reloc .+2, R_390_20, DISP
    lg %r6, 0(%r7,%r8)
