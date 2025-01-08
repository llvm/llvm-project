// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -x c++ -std=c++20 %t/A.cppm -I%t -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 %t/B.cppm -I%t -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 -fprebuilt-module-path=%t %t/foo.cpp -verify
//
// RUN: rm %t/A.pcm %t/B.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 %t/A.cppm -I%t -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 %t/B.cppm -I%t -emit-reduced-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 -fprebuilt-module-path=%t %t/foo.cpp -verify

//--- foo.h
template <class T>
concept A = true;

//--- bar.h
template <class T>
concept A = false;

//--- A.cppm
module;
#include "foo.h"
export module A;
export using ::A;

//--- B.cppm
module;
#include "bar.h"
export module B;
export using ::A;

//--- foo.cpp
import A;
import B;

template <class T> void foo() requires A<T> {}  // expected-error 1+{{reference to 'A' is ambiguous}}
                                                // expected-note@* 1+{{candidate found by name lookup}}

int main() {
  foo<int>();
  return 0;
}
