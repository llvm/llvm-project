# REQUIRES: x86
## Check that section ordering follows from input file ordering.
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/1.s -o %t/1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/2.s -o %t/2.o
# RUN: %lld -dylib %t/1.o %t/2.o -o %t/12
# RUN: %lld -dylib %t/2.o %t/1.o -o %t/21
# RUN: llvm-objdump --macho --section-headers %t/12 | FileCheck %s --check-prefix=CHECK-12
# RUN: llvm-objdump --macho --section-headers %t/21 | FileCheck %s --check-prefix=CHECK-21

# CHECK-12:      __text
# CHECK-12-NEXT: foo
# CHECK-12-NEXT: bar
# CHECK-12-NEXT: __cstring

# CHECK-21:      __text
# CHECK-21-NEXT: __cstring
# CHECK-21-NEXT: bar
# CHECK-21-NEXT: foo

#--- 1.s
.section __TEXT,foo
  .space 1
.section __TEXT,bar
  .space 1
.cstring
  .asciz ""

#--- 2.s
.cstring
  .asciz ""
.section __TEXT,bar
  .space 1
.section __TEXT,foo
  .space 1
