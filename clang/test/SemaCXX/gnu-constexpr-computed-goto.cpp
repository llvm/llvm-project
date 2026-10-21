// RUN: %clang_cc1 -fsyntax-only -verify=beforecxx14,beforecxx23 -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify=beforecxx23 -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify=beforecxx23 -std=c++20  %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

// RUN: %clang_cc1 -fsyntax-only -verify=beforecxx14,beforecxx23 -std=c++11 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fsyntax-only -verify=beforecxx23 -std=c++14 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fsyntax-only -verify=beforecxx23 -std=c++20  %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s -fexperimental-new-constant-interpreter
// expected-no-diagnostics

constexpr int GNUConstexprComputedGoto() {
  return 0; // beforecxx14-note {{previous}}
  goto *(&&x); // beforecxx23-warning {{use of this statement in a constexpr function is a C++23 extension}}
  x:;
    return 0; // beforecxx14-warning {{multiple return}}
}
