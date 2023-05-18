// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 %t/impl.cppm -emit-module-interface -o %t/M-impl.pcm
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/UseStrict.cpp -fprebuilt-module-path=%t -verify -fsyntax-only

//--- impl.cppm
module M:impl;
class A {};

//--- M.cppm
export module M;
import :impl;
export A f();

//--- UseStrict.cpp
import M;
void test() {
  auto a = f(); // expected-error {{definition of 'A' must be imported from module 'M' before it is required}} expected-error{{}}
                // expected-note@* {{definition here is not reachable}} expected-note@* {{}}
}
