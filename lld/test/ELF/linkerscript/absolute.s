# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { foo = ABSOLUTE(.) + 1; };" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o
# RUN: llvm-readobj --symbols %t | FileCheck %s

# CHECK:        Name: foo
# CHECK-NEXT:   Value:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Binding:
# CHECK-NEXT:   Type:
# CHECK-NEXT:   Other:
# CHECK-NEXT:   Section: Absolute
# CHECK-NEXT: }

.text
.globl _start
_start:
