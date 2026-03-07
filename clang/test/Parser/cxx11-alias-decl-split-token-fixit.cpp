// RUN: not %clang_cc1 -fsyntax-only -std=c++11 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template <typename> struct X {};
using A = X<int>>; // expected-error {{expected ';' after alias declaration}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:17-[[@LINE-1]]:17}:";"
