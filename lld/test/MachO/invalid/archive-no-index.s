# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/2.s -o %t/2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/3.s -o %t/3.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/4.s -o %t/4.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/main.o

# RUN: llvm-ar rcS %t/test.a %t/2.o %t/3.o %t/4.o

# RUN: not %lld %t/test.o %t/test.a -o /dev/null 2>&1 | FileCheck %s
# CHECK: error: {{.*}}.a: archive has no index; run ranlib to add one

#--- 2.s
.globl _boo
_boo:
  ret

#--- 3.s
.globl _bar
_bar:
  ret

#--- 4.s
.globl _undefined, _unused
_unused:
  ret

#--- main.s
.global _main
_main:
  mov $0, %rax
  ret
