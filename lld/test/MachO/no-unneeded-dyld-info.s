# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -o %t %t.o
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s

# CHECK:                 cmd LC_DYLD_INFO_ONLY
# CHECK-NEXT:        cmdsize 48
# CHECK-NEXT:     rebase_off 0
# CHECK-NEXT:    rebase_size 0
# CHECK-NEXT:       bind_off 0
# CHECK-NEXT:      bind_size 0
# CHECK-NEXT:  weak_bind_off 0
# CHECK-NEXT: weak_bind_size 0
# CHECK-NEXT:  lazy_bind_off 0
# CHECK-NEXT: lazy_bind_size 0

.globl _main
_main:
  ret
