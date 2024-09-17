# REQUIRES: x86

# RUN: rm -rf %t && mkdir %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t/a.o

# RUN: ld.lld -pie %t/a.o -o %t/out \
# RUN:     --override-section-flags 'foo0=' \
# RUN:     --override-section-flags 'foo1=a' \
# RUN:     --override-section-flags 'foo2=ax' \
# RUN:     --override-section-flags 'foo3=aw' \
# RUN:     --override-section-flags 'foo4=awx'

# RUN: llvm-readelf --sections --segments %t/out | FileCheck %s

# CHECK-DAG: foo0 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}}     {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}}
# CHECK-DAG: foo1 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} A   {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}}
# CHECK-DAG: foo2 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} AX  {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}}
# CHECK-DAG: foo3 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} WA  {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}}
# CHECK-DAG: foo4 PROGBITS {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} WAX {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}}


.globl _start
_start:

.section foo0,"aw"
.space 8

.section foo1,"aw"
.space 8

.section foo2,"aw"
.space 8

.section foo3,"ax"
.space 8

.section foo4,"a"
.space 8
