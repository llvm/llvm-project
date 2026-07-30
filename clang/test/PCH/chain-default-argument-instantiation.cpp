// Test default argument instantiation in chained PCH.

// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -include %s -include %s %s

// With PCH
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -chain-include %s -chain-include %s

// With modules
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -fmodules %s -chain-include %s -chain-include %s

// expected-no-diagnostics

#ifndef HEADER1
#define HEADER1
//===----------------------------------------------------------------------===//
// Primary header.

namespace rdar23810407 {
  template<typename T> int f(T t) {
    extern T rdar23810407_variable;
    return 0;
  }
  template<typename T> int g(int a = f([] {}));
}

//===----------------------------------------------------------------------===//
#elif not defined(HEADER2)
#define HEADER2
#if !defined(HEADER1)
#error Header inclusion order messed up
#endif

//===----------------------------------------------------------------------===//
// Dependent header.

inline void instantiate_once() {
  rdar23810407::g<int>();
}

//===----------------------------------------------------------------------===//
#else
//===----------------------------------------------------------------------===//

void test() {
  rdar23810407::g<int>();
}

//===----------------------------------------------------------------------===//
#endif
