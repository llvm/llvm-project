// RUN: %clang_cc1 -verify -fsyntax-only -Wno-ignored-attributes %s
// expected-no-diagnostics

namespace GH76521 {

template <typename T>
void foo() {
  auto l = []() __attribute__((pcs("aapcs-vfp"))) {};
}

void bar() {
  foo<int>();
}

}
