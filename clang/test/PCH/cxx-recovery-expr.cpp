// Test with pch.
// RUN: %clang_cc1 -emit-pch -frecovery-ast -fallow-pch-with-compiler-errors -o %t %s
// RUN: %clang_cc1 -include-pch %t -fno-validate-pch -emit-llvm -o - %s

#ifndef HEADER
#define HEADER

int func(int);
int s = func();

#else

#endif
