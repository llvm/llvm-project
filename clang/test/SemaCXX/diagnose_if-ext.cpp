// RUN: %clang_cc1 -Wpedantic -fsyntax-only %s -verify

void foo() __attribute__((diagnose_if(1, "", "error"))); // expected-warning{{'diagnose_if' is a clang extension}}
void foo(int a) __attribute__((diagnose_if(a, "", "error"))); // expected-warning{{'diagnose_if' is a clang extension}}
// FIXME: When diagnose_if gets a CXX11 spelling, this should be enabled.
#if 0
[[clang::diagnose_if(a, "", "error")]] void foo(double a);
#endif
