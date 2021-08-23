class A { A(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTBEGIN
class B { B(int i); };
// NOLINTEND

// NOLINTBEGIN
// NOLINTEND
// NOLINTBEGIN
class B1 { B1(int i); };
// NOLINTEND

// NOLINTBEGIN
// NOLINTEND
class B2 { B2(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

// NOLINTBEGIN(google-explicit-constructor)
class C { C(int i); };
// NOLINTEND(google-explicit-constructor)

// NOLINTBEGIN(*)
class C1 { C1(int i); };
// NOLINTEND(*)

// NOLINTBEGIN(some-other-check)
class C2 { C2(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit
// NOLINTEND(some-other-check)

// NOLINTBEGIN(some-other-check, google-explicit-constructor)
class C3 { C3(int i); };
// NOLINTEND(some-other-check, google-explicit-constructor)

// NOLINTBEGIN(some-other-check, google-explicit-constructor)
// NOLINTEND(some-other-check)
class C4 { C4(int i); };
// NOLINTEND(google-explicit-constructor)

// NOLINTBEGIN(some-other-check, google-explicit-constructor)
// NOLINTEND(google-explicit-constructor)
class C5 { C5(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit
// NOLINTEND(some-other-check)

// NOLINTBEGIN(google-explicit-constructor)
// NOLINTBEGIN(some-other-check)
class C6 { C6(int i); };
// NOLINTEND(some-other-check)
// NOLINTEND(google-explicit-constructor)

// NOLINTBEGIN(not-closed-bracket-is-treated-as-skip-all
class C7 { C7(int i); };
// NOLINTEND(not-closed-bracket-is-treated-as-skip-all

// NOLINTBEGIN without-brackets-skip-all, another-check
class C8 { C8(int i); };
// NOLINTEND without-brackets-skip-all, another-check

#define MACRO(X) class X { X(int i); };

MACRO(D1)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit

// NOLINTBEGIN
MACRO(D2)
// NOLINTEND

#define MACRO_NOARG class E { E(int i); };
// NOLINTBEGIN
MACRO_NOARG
// NOLINTEND

// CHECK-MESSAGES: Suppressed 11 warnings (11 NOLINT)

// RUN: %check_clang_tidy %s google-explicit-constructor %t --
