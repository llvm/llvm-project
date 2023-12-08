// RUN: %clang_cc1 -fmodules -std=c++14 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

#pragma clang module build A
module A {}
#pragma clang module contents
#pragma clang module begin A
template<int> int n = 42;
decltype(n<0>) f();
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build B
module B {}
#pragma clang module contents
#pragma clang module begin B
#pragma clang module import A
inline int f() { return n<0>; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import B

// CHECK: @_Z1nILi0EE = linkonce_odr global i32 42, comdat
int g() { return f(); }
