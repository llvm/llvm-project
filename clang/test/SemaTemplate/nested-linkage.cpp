// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

extern "C" { extern "C++" { template<class C> C x(); } }
