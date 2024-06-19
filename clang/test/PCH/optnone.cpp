// RUN: %clang_cc1 -emit-pch -x c++-header %s -o %t.pch
// RUN: %clang_cc1 -emit-llvm -DMAIN -include-pch %t.pch %s -o /dev/null

#ifndef MAIN
__attribute__((optnone)) void foo() {}
#endif
