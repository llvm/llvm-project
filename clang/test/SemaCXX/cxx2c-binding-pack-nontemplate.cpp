// RUN: %clang_cc1 -std=c++26 -fsyntax-only %s -verify=nontemplate
// RUN: %clang_cc1 -std=c++2c -verify=cxx26,nontemplate -fsyntax-only -Wpre-c++26-compat %s
// RUN: %clang_cc1 -std=c++23 -verify=cxx23,nontemplate -fsyntax-only -Wc++26-extensions %s

void decompose_array() {
  int arr[4] = {1, 2, 3, 6};
  // cxx26-warning@+3 {{structured binding packs are incompatible with C++ standards before C++2c}}
  // cxx23-warning@+2 {{structured binding packs are a C++2c extension}}
  // nontemplate-error@+1 {{pack declaration outside of template}}
  auto [x, ...rest, y] = arr;
}
