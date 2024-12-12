// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header header.h
// RUN: %clang_cc1 -std=c++20 -fmodule-file=header.pcm main.cpp

//--- header.h
template <typename T>
void f(T) {}

class A {
  virtual ~A();
};

inline A::~A() {
  f([](){});
}

struct B {
  void g() {
    f([](){
      [](){};
    });
  }
};
// expected-no-diagnostics

//--- main.cpp
import "header.h";
// expected-no-diagnostics
