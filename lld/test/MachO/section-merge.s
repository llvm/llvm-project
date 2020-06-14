# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libhello.s \
# RUN:   -o %t/libhello.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libgoodbye.s \
# RUN:   -o %t/libgoodbye.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libfunction.s \
# RUN:   -o %t/libfunction.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s \
# RUN:   -o %t/main.o
# RUN: lld -flavor darwinnew -o %t/output %t/libfunction.o %t/libgoodbye.o %t/libhello.o %t/main.o

# RUN: llvm-objdump --syms %t/output | FileCheck %s
# CHECK:      SYMBOL TABLE:
# CHECK-DAG:  {{[0-9a-z]+}} g     O __TEXT,__cstring _goodbye_world
# CHECK-DAG:  {{[0-9a-z]+}} g     O __TEXT,__cstring _hello_its_me
# CHECK-DAG:  {{[0-9a-z]+}} g     O __TEXT,__cstring _hello_world
# CHECK-DAG:  {{[0-9a-z]+}} g     F __TEXT,__text _main
# CHECK-DAG:  {{[0-9a-z]+}} g     F __TEXT,__text _some_function

# RUN: llvm-objdump -d %t/output | FileCheck %s --check-prefix DATA
# DATA:      Disassembly of section __TEXT,__text:
# DATA:        {{0*}}[[#%x,BASE:]] <_some_function>:
# DATA-NEXT:             [[#BASE]]: 48 c7 c0 01 00 00 00          movq    $1, %rax
# DATA-NEXT:       [[#BASE + 0x7]]: c3                            retq
# DATA:        {{0*}}[[#%x,MAIN:]] <_main>:
# DATA-NEXT:             [[#MAIN]]: 48 c7 c0 00 00 00 00          movq    $0, %rax
# DATA-NEXT:       [[#MAIN + 0x7]]: c3                            retq

.section __TEXT,__text
.global _main

_main:
  mov $0, %rax
  ret
