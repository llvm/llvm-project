// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/std.cppm -emit-module-interface -o %t/std.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod.cppm -fprebuilt-module-path=%t -emit-module-interface -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -verify %t/main.cpp

//--- std.cppm
export module std;

extern "C++" {
  namespace std {
  export template <class E>
  class initializer_list {
    const E* _1;
    const E* _2;
  };
  }
}

//--- mod.cppm
export module mod;

import std;

export struct A {
  void func(std::initializer_list<int>) {}
};

//--- main.cpp
// expected-no-diagnostics
import std;
import mod;

int main() {
  A{}.func({1,1});
  return 0;
}
