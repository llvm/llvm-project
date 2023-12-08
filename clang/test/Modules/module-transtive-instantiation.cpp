// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang_cc1 -std=c++20 %S/Inputs/module-transtive-instantiation/Templ.cppm -emit-module-interface -o %t/Templ.pcm
// RUN: %clang_cc1 -std=c++20 %S/Inputs/module-transtive-instantiation/bar.cppm  -emit-module-interface -fprebuilt-module-path=%t -o %t/bar.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %s -fsyntax-only -verify
// expected-no-diagnostics

import bar;
int foo() {
  return bar<int>();
}
