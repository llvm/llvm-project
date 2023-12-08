# REQUIRES: x86
# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -o %t/test.pie %t/test.o
# RUN: llvm-objdump --macho --syms %t/test.pie | FileCheck %s --check-prefix=PIE

# RUN: %lld -o %t/test.no_pie %t/test.o -no_pie
# RUN: llvm-objdump --macho --syms %t/test.no_pie | FileCheck %s --check-prefix=NO-PIE

# PIE:    0000000100000000 g     F __TEXT,__text __mh_execute_header
# NO-PIE: 0000000000000000 g       *ABS* __mh_execute_header

.text
.global _main
_main:
  ret
