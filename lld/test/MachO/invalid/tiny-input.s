# REQUIRES: x86

## Check that files too short to have a magic number are rejected as inputs
# RUN: echo -n 1 >%t-1.o
# RUN: echo -n 12 >%t-2.o
# RUN: echo -n 123 >%t-3.o
# RUN: echo -n 1234 >%t-4.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -o %t %t.o %t-1.o %t-2.o %t-3.o %t-4.o 2>&1 | FileCheck %s

# CHECK: error: file is too small to contain a magic number: {{.*}}-1.o
# CHECK: error: file is too small to contain a magic number: {{.*}}-2.o
# CHECK: error: file is too small to contain a magic number: {{.*}}-3.o
# CHECK: error: {{.*}}-4.o: unhandled file type

.global _main
_main:
  ret
