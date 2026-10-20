// RUN: %clang_cc1 -fsyntax-only -verify -std=c++26 -Wunused %s

namespace GH125810 {
struct S {
  int a, b;
};

void t(S s) {
  auto &[_, _] = s;
  auto &[a1, _] = s; // expected-warning {{unused variable '[a1, _]'}}
  auto &[_, b2] = s; // expected-warning {{unused variable '[_, b2]'}}

  auto &[a3 [[maybe_unused]], b3 [[maybe_unused]]] = s;
  auto &[a4, b4 [[maybe_unused]]] = s; // expected-warning {{unused variable '[a4, b4]'}}
  auto &[a5 [[maybe_unused]], b5] = s; // expected-warning {{unused variable '[a5, b5]'}}
}
}
