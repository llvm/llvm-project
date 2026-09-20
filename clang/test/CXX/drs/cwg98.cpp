// RUN: %clang_cc1 -std=c++98 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++11 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++14 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++17 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++20 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++23 %s -fexceptions -fcxx-exceptions -pedantic-errors -verify
// RUN: %clang_cc1 -std=c++2c %s -fexceptions -fcxx-exceptions -pedantic-errors -verify

namespace cwg98 { // cwg98: 2.7
  void test(int n) {
    switch (n) {
      try { // #cwg98-try
        case 0:
        // expected-error@-1 {{cannot jump from switch statement to this case label}}
        //   expected-note@#cwg98-try {{jump bypasses initialization of try block}}
        x:
          throw n;
      } catch (...) { // #cwg98-catch
        case 1:
        // expected-error@-1 {{cannot jump from switch statement to this case label}}
        //   expected-note@#cwg98-catch {{jump bypasses initialization of catch block}}
        y:
          throw n;
      }
      case 2:
        goto x;
        // expected-error@-1 {{cannot jump from this goto statement to its label}}
        //   expected-note@#cwg98-try {{jump bypasses initialization of try block}}
      case 3:
        goto y;
        // expected-error@-1 {{cannot jump from this goto statement to its label}}
        //   expected-note@#cwg98-catch {{jump bypasses initialization of catch block}}
    }
  }
} // namespace cwg98
