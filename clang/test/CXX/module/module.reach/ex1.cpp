// From [module.reach]p4, example 1
//
// RUN: rm -fr %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/M-A.cppm -o %t/M-A.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fprebuilt-module-path=%t %t/M-B-impl.cppm -o %t/M-B.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/M-C.cppm -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fprebuilt-module-path=%t %t/M.cppm -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/X.cppm -fsyntax-only -verify
//
//--- M-A.cppm
export module M:A;
export struct B;

//--- M-B-impl.cppm
module M:B;
struct B {
  operator int();
};

//--- M-C.cppm
module M:C;
import :A;
B b1; // expected-error {{variable has incomplete type 'B'}}
      // expected-note@* {{forward declaration of 'B'}}

//--- M.cppm
export module M;
export import :A;
import :B;
B b2;
export void f(B b = B());

//--- X.cppm
export module X;
import M;
B b3; // expected-error {{definition of 'B' must be imported from module 'M' before it is required}} expected-error {{}}
      // expected-note@* {{definition here is not reachable}} expected-note@* {{}}
// FIXME: We should emit an error for unreachable definition of B.
void g() { f(); }
void g1() { f(B()); } // expected-error 1+{{definition of 'B' must be imported from module 'M' before it is required}}
                      // expected-note@* 1+{{definition here is not reachable}}
                      // expected-note@M.cppm:5 {{passing argument to parameter 'b' here}}
