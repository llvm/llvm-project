// RUN: %clang_cc1 -emit-llvm-only -fmodules -triple x86_64-windows %s
// PR36181
#pragma clang module build foo
module foo {}
#pragma clang module contents
template <typename T> struct A {
  friend void f(A<T>) {}
};
#pragma clang module endbuild
#pragma clang module import foo
void g() { f(A<int>()); }
