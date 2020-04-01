# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-readobj --macho-segment %t | FileCheck %s

# CHECK: Name: __TEXT
# CHECK-NOT: }
# dyld3 assumes that the __TEXT segment starts from the file header
# CHECK:       fileoff: 0

.text
.global _main
_main:
  mov $0, %rax
  ret
