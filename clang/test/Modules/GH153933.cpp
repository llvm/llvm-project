// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fprebuilt-module-path=%t %t/C.cpp

//--- A.hpp
template<class> struct A {};
template<class T> struct B {
  virtual A<T> v() { return {}; }
};
B<void> x;

//--- B.cppm
module;
#include "A.hpp"
export module B;
using ::x;

//--- C.cpp
#include "A.hpp"
import B;
