// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify -fcxx-exceptions %s
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify=good -fcxx-exceptions %s
// RUN: %clang_cc1 -fsyntax-only -Wno-invalid-constexpr -verify=good -fcxx-exceptions %s
// good-no-diagnostics

constexpr void func() { // expected-error {{constexpr function never produces a constant expression}}
  throw 12;             // expected-note {{subexpression not valid in a constant expression}}
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winvalid-constexpr"
constexpr void other_func() {
#pragma clang diagnostic pop

  throw 12;
}
