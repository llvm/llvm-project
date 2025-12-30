// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

namespace A {
  namespace B { }
}

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:7-[[@LINE+1]]:7}:"namespace "
using A::B; // expected-error{{using declaration cannot refer to a namespace}}
            // expected-note@-1 {{did you mean 'using namespace'?}}
