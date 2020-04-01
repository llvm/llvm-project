# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-readobj --macho-segment %t | FileCheck %s

# These segments must always be present.
# CHECK-DAG: Name: __PAGEZERO
# CHECK-DAG: Name: __LINKEDIT
# CHECK-DAG: Name: __TEXT

# Check that we handle max-length names correctly.
# CHECK-DAG: Name: maxlen_16ch_name

.text
.global _main
_main:
  mov $0, %rax
  ret

.section maxlen_16ch_name,foo
