// RUN: %clang_cc1 -triple mips-none-none -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=FP64
// RUN: %clang_cc1 -triple powerpc64-none-none -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=FP128

template <class T> void g3(char (&buffer)[sizeof(T() + 5.0)]) {}
template void g3<int>(char (&)[sizeof(double)]);
// CHECK: _Z2g3IiEvRAszplcvT__ELd4014000000000000E_c

template <class T> void g4(char (&buffer)[sizeof(T() + 5.0L)]) {}
template void g4<int>(char (&)[sizeof(long double)]);
// FP64: _Z2g4IiEvRAszplcvT__ELe4014000000000000E_c
// FP128: _Z2g4IiEvRAszplcvT__ELg00000000000000004014000000000000E_c

template <class T> void g5(char (&buffer)[sizeof(T() + 5)]) {}
template void g5<int>(char (&)[sizeof(int)]);
// CHECK: _Z2g5IiEvRAszplcvT__ELi5E_c

template <class T> void g6(char (&buffer)[sizeof(T() + 5L)]) {}
template void g6<int>(char (&)[sizeof(long int)]);
// CHECK: _Z2g6IiEvRAszplcvT__ELl5E_c
