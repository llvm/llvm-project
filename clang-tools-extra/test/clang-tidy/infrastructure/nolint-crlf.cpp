// RUN: %check_clang_tidy %s google-explicit-constructor %t

class A { A(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

class B { B(int i); }; // NOLINT

// NOLINTNEXTLINE
class C { C(int i); };

// NOLINTBEGIN
class D { D(int i); };
// NOLINTEND

const char *S = "NOLINT";
class E { E(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

const char *SNL = "NOLINTNEXTLINE";
class F { F(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// CHECK-MESSAGES: Suppressed 3 warnings (3 NOLINT)
