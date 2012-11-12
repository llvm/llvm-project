// RUN: %clang_cc1 -std=gnu89 %s -emit-llvm -o - | not grep dead_function

extern __inline__ void dead_function() {}
