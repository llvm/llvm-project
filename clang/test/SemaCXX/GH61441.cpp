// RUN: %clang_cc1 -fsyntax-only -verify -cl-std=clc++ -fblocks %s
// Checks Clang does not crash. We run in OpenCL mode to trigger block pointer
// crash. The __fp16 crash happens in standard mode too.

template <bool>
int foo() {
  auto x = [&](__fp16) { return 0; };       // expected-error {{not allowed}}
  auto y = [&](void(^)(int)) { return 0; }; // expected-error {{not allowed}}
  return 0;
}

int bar() { return foo<true>(); }
