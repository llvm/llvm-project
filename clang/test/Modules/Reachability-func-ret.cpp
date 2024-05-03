// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/func_ret.cppm -emit-module-interface -o %t/func_ret.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only

// RUN: %clang_cc1 -std=c++20 %t/func_ret.cppm -emit-reduced-module-interface -o %t/func_ret.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only
//
//--- func_ret.cppm
export module func_ret;
struct t {};
export t foo() {
  return t{};
}

//--- Use.cpp
// expected-no-diagnostics
import func_ret;
void bar() {
  auto ret = foo();
}
