// RUN: %clang_cc1 -emit-llvm %s -o -
struct A {
  unsigned long long : (sizeof(unsigned long long) * 8) - 16;
};
struct B {
  A a;
};
struct B b = {
  {}
};
