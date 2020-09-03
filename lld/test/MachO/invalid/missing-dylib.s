# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not lld -flavor darwinnew -Z -o %t -lmissing %t.o 2>&1 | FileCheck %s

# CHECK: error: library not found for -lmissing
