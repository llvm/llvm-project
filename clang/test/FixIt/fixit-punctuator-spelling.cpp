// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

void f(int x) {
  switch (x) {
    case 1 // expected-error {{expected ':' after 'case'}}
      break;
  }
}
// CHECK: fix-it:"{{.*}}":{6:11-6:11}:":"
