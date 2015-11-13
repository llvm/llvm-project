# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: not ld.lld2 -r %t -o %t2 2>&1 | FileCheck %s

# CHECK: -r option is not supported. Use 'ar' command instead.

.globl _start;
_start:
