# REQUIRES: x86
# RUN: mkdir -p %t
# RUN: echo ".global _boo; _boo: ret"                           | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/2.o
# RUN: echo ".global _bar; _bar: ret"                           | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/3.o
# RUN: echo ".global _undefined; .global _unused; _unused: ret" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/4.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/main.o

# RUN: rm -f %t/test.a
# RUN: llvm-ar rcS %t/test.a %t/2.o %t/3.o %t/4.o

# RUN: not lld -flavor darwinnew %t/test.o %t/test.a -o /dev/null 2>&1 | FileCheck %s
# CHECK: error: {{.*}}.a: archive has no index; run ranlib to add one

.global _main
_main:
  mov $0, %rax
  ret
