# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386-apple-darwin %s -o %t.i386.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.x86_64.o
# RUN: llvm-lipo %t.i386.o %t.x86_64.o -create -o %t.fat.o
# RUN: %lld -o /dev/null %t.fat.o

# RUN: llvm-lipo %t.i386.o -create -o %t.noarch.o
# RUN: not %lld -o /dev/null %t.noarch.o 2>&1 | \
# RUN:    FileCheck %s -DFILE=%t.noarch.o
# CHECK: error: unable to find matching architecture in [[FILE]]

.text
.global _main
_main:
  mov $0, %eax
  ret
