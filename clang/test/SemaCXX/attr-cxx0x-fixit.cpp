// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -std=c++11 %s 2>&1 | FileCheck %s

[[noreturn()]] void f(); // expected-error {{attribute 'noreturn' cannot have an argument list}} \
// CHECK: fix-it:"{{.*}}":{4:11-4:13}:""
