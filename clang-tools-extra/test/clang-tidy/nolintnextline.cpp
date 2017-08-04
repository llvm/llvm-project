class A { A(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE
class B { B(int i); };

// NOLINTNEXTLINE(we-dont-care-about-categories-yet)
class C { C(int i); };


// NOLINTNEXTLINE

class D { D(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE
//
class E { E(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

#define MACRO(X) class X { X(int i); };
MACRO(F)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit
// NOLINTNEXTLINE
MACRO(G)

#define MACRO_NOARG class H { H(int i); };
// NOLINTNEXTLINE
MACRO_NOARG

// CHECK-MESSAGES: Suppressed 4 warnings (4 NOLINT)

// RUN: %check_clang_tidy %s google-explicit-constructor %t --
