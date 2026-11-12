// RUN: clang-check -analyze "%s" -- -c 2>&1 | FileCheck %s
// RUN: clang-check -analyze "%s" -- -c -flto -Wa,--noexecstack 2>&1 | FileCheck %s
// RUN: clang-check -analyze "%s" -- -c -no-integrated-as -flto=thin 2>&1 | FileCheck %s
// RUN: clang-check -analyze "%s" -- -c -flto=full 2>&1 | FileCheck %s

// CHECK: Dereference of null pointer
void a(int *x) { if(x){} *x = 47; }
