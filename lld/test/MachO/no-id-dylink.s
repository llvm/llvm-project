# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: yaml2obj %p/Inputs/no-id-dylib.yaml -o %t/libnoid.dylib
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/no-id-dylink.o
# RUN: not lld -flavor darwinnew -o %t/no-id-dylink -Z -L%t -lnoid %t/no-id-dylink.o 2>&1 | FileCheck %s
# CHECK: dylib {{.*}}libnoid.dylib missing LC_ID_DYLIB load command

.text
.globl _main

_main:
  mov $0, %rax
  ret
