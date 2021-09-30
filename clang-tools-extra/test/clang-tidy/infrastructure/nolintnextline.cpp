// NOLINTNEXTLINE
class A { A(int i); };

class B { B(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE
class C { C(int i); };

// NOLINTNEXTLINE(for-some-other-check)
class D { D(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE(*)
class D1 { D1(int i); };

// NOLINTNEXTLINE(not-closed-bracket-is-treated-as-skip-all
class D2 { D2(int i); };

// NOLINTNEXTLINE(google-explicit-constructor)
class D3 { D3(int i); };

// NOLINTNEXTLINE(some-check, google-explicit-constructor)
class D4 { D4(int i); };

// NOLINTNEXTLINE without-brackets-skip-all, another-check
class D5 { D5(int i); };

// NOLINTNEXTLINE

class E { E(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE
//
class F { F(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

#define MACRO(X) class X { X(int i); };
MACRO(G)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit
// NOLINTNEXTLINE
MACRO(H)

#define MACRO_NOARG class I { I(int i); };
// NOLINTNEXTLINE
MACRO_NOARG

// CHECK-MESSAGES: Suppressed 9 warnings (9 NOLINT)

// RUN: %check_clang_tidy %s google-explicit-constructor %t --
