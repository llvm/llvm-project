// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions
// expected-no-diagnostics

void f() throw(...) { }
