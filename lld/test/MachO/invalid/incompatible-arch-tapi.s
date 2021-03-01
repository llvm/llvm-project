# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macosx %t/main.s -o %t/main.o
# RUN: not %lld -arch x86_64 -lSystem %S/Inputs/libincompatible.tbd %t/main.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARCH
# ARCH: error: {{.*}}libincompatible.tbd is incompatible with x86_64

#--- main.s
.globl _main
_main:
  ret
