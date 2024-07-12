// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace GH84473_bug {
void f1() {
  int b;
  (void) [=] [[gnu::regcall]] () {
    (void) b;
  };
}
}
