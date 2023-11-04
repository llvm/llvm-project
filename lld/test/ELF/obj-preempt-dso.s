# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld -shared --version-script=b.ver b.o -o b.so
# RUN: ld.lld --version-script=a.ver a.o b.so -o a
# RUN: llvm-readelf --dyn-syms a | FileCheck %s

# CHECK:      1: 0000000000000000 0 NOTYPE  GLOBAL DEFAULT   UND b2
# CHECK-NEXT: 2: {{.*}}           0 NOTYPE  GLOBAL DEFAULT [[#]] a1
# CHECK-NEXT: 3: {{.*}}           0 NOTYPE  GLOBAL DEFAULT [[#]] a2
# CHECK-NEXT: 4: {{.*}}           0 NOTYPE  GLOBAL DEFAULT [[#]] b1
# CHECK-NEXT: 5: {{.*}}           0 NOTYPE  GLOBAL DEFAULT [[#]] c2@@v2
# CHECK-NEXT: 6: {{.*}}           0 NOTYPE  GLOBAL DEFAULT [[#]] c1@v1
# CHECK-NOT:  {{.}}

#--- a.s
.globl _start, a1, a2, a3, b1, c1, c2
_start:
a1: a2:
.hidden a3
a3:
b1:
.symver c1, c1@v1, remove
c1:
c2:

.data
  .quad b2

#--- a.ver
v1 {};
v2 { c2; };

#--- b.s
.globl a1, a2, a3, b1, b2, c1, c2
.type a1,@function
a1: a2: a3:
.protected b1, b2
b1:
b2:
.symver c1, c1@v1
c1:
c2:

#--- b.ver
v1 {};
v2 { c2; };
