// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify=cxx26 -Wpre-c++26-compat
// RUN: %clang_cc1 %s -std=c++23 -fsyntax-only -verify=cxx23

void f() {
  template for (auto _ : {1}) { // cxx23-warning {{expansion statements are a C++2c extension}} \
                                // cxx26-warning {{expansion statements are incompatible with C++ standards before C++2c}}
  }
}
