// RUN: %clang_cc1 -verify -std=c++2c %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -std=c++2c -fixit %t
// RUN: %clang_cc1 -x c++ -std=c++2c %t
// RUN: not %clang_cc1 -std=c++2c -x c++ -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

static_assert(true, L""); // expected-error{{an unevaluated string literal cannot have an encoding prefix}}
// CHECK: fix-it:{{.*}}:{7:21-7:22}

static_assert(true, u8""); // expected-error{{an unevaluated string literal cannot have an encoding prefix}}
// CHECK: fix-it:{{.*}}:{10:21-10:23}

static_assert(true, u""); // expected-error{{an unevaluated string literal cannot have an encoding prefix}}
// CHECK: fix-it:{{.*}}:{13:21-13:22}

static_assert(true, U""); // expected-error{{an unevaluated string literal cannot have an encoding prefix}}
// CHECK: fix-it:{{.*}}:{16:21-16:22}
