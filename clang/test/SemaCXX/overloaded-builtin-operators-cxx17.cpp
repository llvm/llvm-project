// RUN: %clang_cc1 -fsyntax-only -fshow-overloads=best -verify -triple x86_64-linux-gnu -std=c++17 %s

struct BoolRef {
  operator bool&();
};

void foo(BoolRef br) {
  // C++ [over.built]p3: Increment for bool was removed in C++17.
  bool b = br++; // expected-error{{cannot increment value of type 'BoolRef'}}
}