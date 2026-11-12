// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -verify -fsyntax-only %s
// expected-no-diagnostics

namespace GH76521 {

template <typename T>
void foo() {
  auto l = []() __attribute__((preserve_most)) {};
}

void bar() {
  foo<int>();
}

}
