// RUN: %clang_cc1 %s -emit-llvm -std=c++11 -o %t

struct A {
  ~A();
};

struct B {
  A a;
};

struct C {
  union {
    B b;
  };

  ~C() noexcept;
};

C::~C() noexcept {}
