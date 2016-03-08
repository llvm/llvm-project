// RUN: %clang_cc1 -std=c++1z -verify %s

namespace std_example {
  struct [[nodiscard]] error_info{
    // ...
  };

  error_info enable_missile_safety_mode();
  void launch_missiles();
  void test_missiles() {
    enable_missile_safety_mode(); // expected-warning {{ignoring return value of function declared with 'nodiscard'}}
    launch_missiles();
  }

  error_info &foo();
  void f() { foo(); } // no warning
}
