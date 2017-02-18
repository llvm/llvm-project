# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS { . = 0x20; . = 0x10; }" > %t.script
# RUN: not ld.lld %t.o --script %t.script -o %t -shared 2>&1 | FileCheck %s
# CHECK: unable to move location counter backward
