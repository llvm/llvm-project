# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -o %t %t.o -L%S/Inputs -lallowable_client 2>&1 | FileCheck %s --check-prefix=NOTALLOWED1
# RUN: not %lld -o %t %t.o -L%S/Inputs -lallowable_client -client_name notallowed 2>&1 | FileCheck %s --check-prefix=NOTALLOWED2
# RUN: %lld -o %t %t.o -L%S/Inputs -lallowable_client -client_name allowed
# RUN: %lld -o %t %t.o -L%S/Inputs -lallowable_client -client_name all

# NOTALLOWED1: error: cannot link directly with 'liballowable_client.dylib' because {{.*}} is not an allowed client
# NOTALLOWED2: error: cannot link directly with 'liballowable_client.dylib' because notallowed is not an allowed client

.text
.global _main
_main:
  mov $0, %rax
  ret
