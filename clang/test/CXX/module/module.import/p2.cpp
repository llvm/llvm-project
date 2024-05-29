// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 %t/impl.cppm -emit-module-interface -o %t/M-impl.pcm
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/Use.cpp -fprebuilt-module-path=%t -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/UseInPartA.cppm -fprebuilt-module-path=%t -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/UseInAnotherModule.cppm -fprebuilt-module-path=%t -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/Private.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/TryUseFromPrivate.cpp -fprebuilt-module-path=%t -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/Global.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/C.pcm
// RUN: %clang_cc1 -std=c++20 %t/UseGlobal.cpp -fprebuilt-module-path=%t -verify -fsyntax-only

//--- impl.cppm
module M:impl;
class A {};

//--- M.cppm
export module M;
import :impl;
export A f();

//--- Use.cpp
import M;
void test() {
  A a; // expected-error {{definition of 'A' must be imported from module 'M' before it is required}}
       // expected-error@-1 {{definition of 'A' must be imported from module 'M' before it is required}} expected-error@-1 {{}}
       // expected-note@impl.cppm:2 {{declaration here is not visible}}
       // expected-note@impl.cppm:2 {{definition here is not reachable}} expected-note@impl.cppm:2 {{}}
}

//--- UseInPartA.cppm
// expected-no-diagnostics
export module M:partA;
import :impl;
void test() {
  A a;
}

//--- UseInAnotherModule.cppm
export module B;
import M;
void test() {
  A a; // expected-error {{declaration of 'A' must be imported from module 'M'}}
       // expected-error@-1 {{definition of 'A' must be imported from module 'M'}} expected-error@-1 {{}}
       // expected-note@impl.cppm:2 {{declaration here is not visible}}
       // expected-note@impl.cppm:2 {{definition here is not reachable}} expected-note@impl.cppm:2 {{}}
}

//--- Private.cppm
export module A;
module :private;

class A {};
void test() {
  A a;
}

//--- TryUseFromPrivate.cpp

import A;
void test() {
  A a; // expected-error {{unknown type name 'A'}}
}

//--- Global.cppm
module;
class A{};
export module C;
void test() {
  A a;
}

//--- UseGlobal.cpp
import C;
void test() {
  A a; // expected-error {{'A' must be declared before it is used}}
       // expected-note@Global.cppm:2 {{declaration here is not visible}}
}
