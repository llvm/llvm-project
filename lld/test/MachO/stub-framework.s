# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: lld -flavor darwinnew -o %t/test -Z -F%S/Inputs/MacOSX.sdk/System/Library/Frameworks -framework CoreFoundation %t/test.o
#
# RUN: llvm-objdump --macho --dylibs-used %t/test | FileCheck %s
# CHECK: /System/Library/Frameworks/CoreFoundation.framework/CoreFoundation

.section __TEXT,__text
.global _main

_main:
  movq __CFBigNumGetInt128@GOTPCREL(%rip), %rax
  ret
