# REQUIRES: x86
## When a common symbol is merged with a shared symbol, pick the larger st_size.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.globl com; .comm com, 16' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld -shared %t1.o -o %t1.so

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readelf -s %t | FileCheck %s
# RUN: ld.lld %t1.so %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s

# CHECK: 16 OBJECT GLOBAL DEFAULT 7 com

.globl com
.comm com,1
