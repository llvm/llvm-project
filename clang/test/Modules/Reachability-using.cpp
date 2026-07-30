// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mod.cppm -emit-module-interface -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/mod.cppm -emit-reduced-module-interface -o %t/mod.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify
//
//--- mod.cppm
export module mod;
struct t {};
export using u = t;

//--- Use.cpp
// expected-no-diagnostics
import mod;
void foo() {
  u v{};
}
