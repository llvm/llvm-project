# REQUIRES: x86
## When a common symbol is merged with a shared symbol, pick the larger st_size.

# RUN: split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 small.s -o small.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 big.s -o big.o
# RUN: ld.lld -shared big.o -o big.so
# RUN: ld.lld -shared small.o -o small.so

## Common arrives first, then larger shared.
# RUN: ld.lld small.o big.so -o out1
# RUN: llvm-readelf -s out1 | FileCheck %s
## Larger shared first, then common (overwrite path, size > this->size).
# RUN: ld.lld big.so small.o -o out2
# RUN: llvm-readelf -s out2 | FileCheck %s
## Smaller shared first, then larger common (overwrite path, size <= this->size).
# RUN: ld.lld small.so big.o -o out3
# RUN: llvm-readelf -s out3 | FileCheck %s

# CHECK: 16 OBJECT GLOBAL DEFAULT [[#]] com

#--- small.s
.globl com
.comm com,1

#--- big.s
.globl com
.comm com,16
