// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s
// expected-no-diagnostics

template <typename T> static void destroy() {
    T t;
    ++t;
}

struct Incomplete;

template <typename = int> struct HasD {
  ~HasD() { destroy<Incomplete*>(); }
};

struct HasVT {
  virtual ~HasVT();
};

struct S : HasVT {
  HasD<> v;
};

