// RUN: %clang_cc1 -std=c++11 -verify=ext -fsyntax-only -Wfuture-attribute-extensions %s
// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only -Wno-future-attribute-extensions %s
// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only -Wno-c++14-attribute-extensions -Wno-c++17-attribute-extensions -Wno-c++20-attribute-extensions %s

// expected-no-diagnostics

[[deprecated]] int func1(); // ext-warning {{use of the 'deprecated' attribute is a C++14 extension}}
[[deprecated("msg")]] int func2(); // ext-warning {{use of the 'deprecated' attribute is a C++14 extension}}

[[nodiscard]] int func3(); // ext-warning {{use of the 'nodiscard' attribute is a C++17 extension}}
[[nodiscard("msg")]] int func4(); // ext-warning {{use of the 'nodiscard' attribute is a C++20 extension}}

void func5() {
  if (true) [[likely]]; // ext-warning {{use of the 'likely' attribute is a C++20 extension}}
}
