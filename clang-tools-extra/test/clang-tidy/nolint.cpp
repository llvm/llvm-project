// RUN: %check_clang_tidy %s google-explicit-constructor,clang-diagnostic-unused-variable %t -- -extra-arg=-Wunused-variable --

class A { A(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

class B { B(int i); }; // NOLINT

class C { C(int i); }; // NOLINT(we-dont-care-about-categories-yet)

void f() {
  int i;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unused variable 'i' [clang-diagnostic-unused-variable]
  int j; // NOLINT
}

// CHECK-MESSAGES: Suppressed 3 warnings (3 NOLINT)
