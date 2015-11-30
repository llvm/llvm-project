// RUN: %check_clang_tidy %s google-explicit-constructor %t

class A { A(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

class B { B(int i); }; // NOLINT

class C { C(int i); }; // NOLINT(we-dont-care-about-categories-yet)
// CHECK-MESSAGES: Suppressed 2 warnings (2 NOLINT)
