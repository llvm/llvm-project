// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/func_default_arg.cppm -emit-module-interface -o %t/func_default_arg.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only

// RUN: %clang_cc1 -std=c++20 %t/func_default_arg.cppm -emit-reduced-module-interface -o %t/func_default_arg.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only
//
//--- func_default_arg.cppm
export module func_default_arg;
struct t {};
export t foo(t t1 = t()) {
  return t1;
}

//--- Use.cpp
// expected-no-diagnostics
import func_default_arg;
void bar() {
  auto ret = foo();
}
