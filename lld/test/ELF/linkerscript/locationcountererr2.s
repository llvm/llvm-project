# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: echo "SECTIONS {" > %t.script
# RUN: echo ". = 0x20; . = 0x10; .text : {} }" >> %t.script
# RUN: not ld.lld %t.o --script %t.script -o %t -shared 2>&1 | FileCheck %s
# CHECK: {{.*}}.script:2: unable to move location counter backward

# RUN: echo "SECTIONS { . = 0x20; . = ASSERT(0x1, "foo"); }" > %t2.script
# RUN: ld.lld %t.o --script %t2.script -o %t -shared
