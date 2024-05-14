// RUN: %clang_cc1 -emit-pch -DHEADER -x c++-header %s -o %t.pch
// RUN: %clang_cc1 -emit-llvm -include-pch %t.pch %s -o /dev/null

#ifdef HEADER
__attribute__((optnone)) void foo() {}
#endif