# REQUIRES: systemz
# RUN: llvm-mc -filetype=obj -triple=s390x -defsym DISP=291 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=s390x -defsym DISP=4095 %s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=s390x -defsym DISP=4096 %s -o %t3.o

# RUN: ld.lld --section-start=.text=0x0 %t1.o -o %t1out
# RUN: ld.lld --section-start=.text=0x0 %t2.o -o %t2out
# RUN: not ld.lld --section-start=.text=0x0 %t3.o -o /dev/null 2>&1 | FileCheck %s --check-prefix RANGE

# RANGE: relocation R_390_12 out of range: 4096 is not in [0, 4095]

# RUN: llvm-readelf --hex-dump=.text %t1out | FileCheck %s -DINSN=58678123 --check-prefix DUMP
# RUN: llvm-readelf --hex-dump=.text %t2out | FileCheck %s -DINSN=58678fff --check-prefix DUMP

# DUMP:  0x00000000 [[INSN]]

.text
.globl _start
_start:
    .reloc .+2, R_390_12, DISP
    l %r6, 0(%r7,%r8)
