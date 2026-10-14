// RUN: not %clang_cc1 -fsyntax-only -std=c++11 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

template <typename> struct X {};
using A = X<int>>;

// CHECK: error: expected ';' after alias declaration
// CHECK: fix-it:"{{.*}}":{4:17-4:17}:";"
