// RUN: %clang_cc1 -verify=ms-ext -fms-extensions -fsyntax-only -std=c++23 %s
// RUN: %clang_cc1 -verify=no-ms-ext -fsyntax-only -std=c++23 %s

void foo();

[[msvc::forceinline]] void func() {}
// no-ms-ext-warning@-1 {{'msvc::forceinline' attribute ignored}}

void lambda_func() {
  auto l = [] [[msvc::forceinline]] () {};
  // no-ms-ext-warning@-1 {{'msvc::forceinline' attribute ignored}}
  l();
}

void stmt_forceinline() {
  [[msvc::forceinline]] func();
  // ms-ext-warning@-1 {{attribute is ignored on this statement as it only applies to functions; use '[[msvc::forceinline_calls]]' on statements}}
  // no-ms-ext-warning@-2 {{'msvc::forceinline' attribute ignored}}
}

[[msvc::forceinline_calls]] void func2() {}
// ms-ext-warning@-1 {{attribute is ignored on this function as it only applies to statements; use '[[msvc::forceinline]]' for functions}}
// no-ms-ext-warning@-2 {{'msvc::forceinline_calls' attribute ignored}}

void stmt_forceinline_calls() {
  [[msvc::forceinline_calls]] foo();
  // no-ms-ext-warning@-1 {{'msvc::forceinline_calls' attribute ignored}}
}

[[msvc::forceinline(0)]] void func3();
// ms-ext-error@-1 {{'msvc::forceinline' attribute takes no arguments}}
// no-ms-ext-error@-2 {{'msvc::forceinline' attribute takes no arguments}}

void func4() {
  [[msvc::forceinline_calls("foo")]] foo();
  // ms-ext-error@-1 {{'msvc::forceinline_calls' attribute takes no arguments}}
  // no-ms-ext-error@-2 {{'msvc::forceinline_calls' attribute takes no arguments}}
}
