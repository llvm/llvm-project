// RUN: %clang_cc1 %s -fcxx-exceptions -fsyntax-only -Wexceptions -verify -std=c++14

// expected-no-diagnostics
struct Base {
  __attribute__((nothrow)) Base() {}
};

struct Derived : Base {
  Derived() noexcept = default;
};

struct Base2 {
   Base2() noexcept {}
};

struct Derived2 : Base2 {
  __attribute__((nothrow)) Derived2() = default;
};

struct Base3 {
  __attribute__((nothrow)) Base3() {}
};

struct Derived3 : Base3 {
  __attribute__((nothrow)) Derived3() = default;
};
