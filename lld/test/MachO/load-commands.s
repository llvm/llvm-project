# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: obj2yaml %t | FileCheck %s

# Check for the presence of a couple of load commands that are essential for
# a working binary.

# CHECK-DAG: cmd:             LC_DYLD_INFO_ONLY
# CHECK-DAG: cmd:             LC_SYMTAB
# CHECK-DAG: cmd:             LC_DYSYMTAB

.text
.global _main
_main:
  mov $0, %rax
  ret
