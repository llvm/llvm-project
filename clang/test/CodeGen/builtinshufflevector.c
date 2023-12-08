// RUN: %clang_cc1 -emit-llvm < %s | grep 'shufflevector' | count 1
// RUN: %clang_cc1 -emit-llvm -ftrapv < %s | grep 'shufflevector' | count 1
typedef int v4si __attribute__ ((vector_size (16)));

v4si a(v4si x, v4si y) {return __builtin_shufflevector(x, y, 3, 2, 5, (2*3)+1);}

