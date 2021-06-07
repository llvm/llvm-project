# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -o %t %t.o

## Check that -rpath generates LC_RPATH.
# RUN: %lld -o %t %t.o -rpath /some/rpath -rpath /another/rpath
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s
# CHECK:      LC_RPATH
# CHECK-NEXT: cmdsize 24
# CHECK-NEXT: path /some/rpath
# CHECK:      LC_RPATH
# CHECK-NEXT: cmdsize 32
# CHECK-NEXT: path /another/rpath

.text
.global _main
_main:
  mov $0, %rax
  ret
