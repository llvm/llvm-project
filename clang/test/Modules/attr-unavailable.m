// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/attr-unavailable %s -fsyntax-only -verify

@import two;
void f(id x) {
  [x method1];
}

@import oneA;
void g(id x) {
  [x method2]; // expected-error{{'method2' is unavailable}}
               // expected-note@oneA.h:2 {{'method2' has been explicitly marked unavailable here}}
  [x method3]; // expected-error{{'method3' is unavailable}}
               // expected-note@oneA.h:3 {{'method3' has been explicitly marked unavailable here}}
}

@import oneB;
void h(id x) {
  [x method2]; // could be from interface D in module oneB
}

@import oneC;
void i(id x) {
  [x method3]; // could be from interface E in module oncC
}
