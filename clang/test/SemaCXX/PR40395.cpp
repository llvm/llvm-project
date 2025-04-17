// RUN: %clang_cc1 -std=c++17 -fms-extensions -triple=x86_64-pc-win32 -verify %s
// RUN: %clang_cc1 -std=c++17 -fms-extensions -triple=x86_64-pc-win32 -verify %s -fexperimental-new-constant-interpreter
// expected-no-diagnostics

// PR40395 - ConstantExpr shouldn't cause the template object to infinitely
// expand.
struct _GUID {};
struct __declspec(uuid("{AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA}")) B {};

template <const _GUID* piid>
struct A {
  virtual void baz() { A<piid>(); }
};

void f() {
  A<&__uuidof(B)>();
}
