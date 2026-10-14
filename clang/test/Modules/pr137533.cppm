// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-module-interface -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=m=%t/m.pcm -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=m=%t/m.pcm -fsyntax-only -verify

//--- m.cppm
export module m;
static const char * f() { return "m.cppm"; }
export auto &fr = f;

//--- use.cpp
// expected-no-diagnostics
import m;
static const char * f() { return "use.cpp"; }
constexpr auto &fr1 = f;
auto &fr2 = fr;
int main() {
  fr1();
  fr2();
}
