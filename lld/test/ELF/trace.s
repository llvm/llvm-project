# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.bar.o

## Check -t
# RUN: ld.lld -shared %t.foo.o -o %t.so -t 2>&1 | FileCheck %s
# CHECK: {{.*}}.foo.o

## Check --trace alias
# RUN: ld.lld -shared %t.foo.o -o %t.so -t 2>&1 | FileCheck %s

## Check output messages order (1)
# RUN: ld.lld -shared %t.foo.o %t1.bar.o -o %t.so -t 2>&1 | FileCheck -check-prefix=ORDER1 %s
# ORDER1: {{.*}}.foo.o
# ORDER1: {{.*}}.bar.o

## Check output messages order (2)
# RUN: ld.lld -shared %t1.bar.o %t.foo.o -o %t.so -t 2>&1 | FileCheck -check-prefix=ORDER2 %s
# ORDER2: {{.*}}.bar.o
# ORDER2: {{.*}}.foo.o
