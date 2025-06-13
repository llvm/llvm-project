// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -fdiagnostics-parseable-fixits -std=c++23 %s 2>&1 | FileCheck %s

#define C(x) case x

void t1(int a) {
  switch (a) {
    C(10) // expected-error {{expected ':' after 'case'}}
  }
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:":"
