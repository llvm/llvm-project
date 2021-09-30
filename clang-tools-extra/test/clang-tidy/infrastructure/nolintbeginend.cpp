// RUN: %check_clang_tidy %s google-explicit-constructor %t

class A { A(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTBEGIN
class B1 { B1(int i); };
// NOLINTEND

// NOLINTBEGIN
// NOLINTEND
// NOLINTBEGIN
class B2 { B2(int i); };
// NOLINTEND

// NOLINTBEGIN
// NOLINTBEGIN
class B3 { B3(int i); };
// NOLINTEND
// NOLINTEND

// NOLINTBEGIN
// NOLINTBEGIN
// NOLINTEND
class B4 { B4(int i); };
// NOLINTEND

// NOLINTBEGIN
// NOLINTEND
class B5 { B5(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

// NOLINTBEGIN(google-explicit-constructor)
class C1 { C1(int i); };
// NOLINTEND(google-explicit-constructor)

// NOLINTBEGIN(*)
class C2 { C2(int i); };
// NOLINTEND(*)

// NOLINTBEGIN(some-other-check)
class C3 { C3(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit
// NOLINTEND(some-other-check)

// NOLINTBEGIN(some-other-check, google-explicit-constructor)
class C4 { C4(int i); };
// NOLINTEND(some-other-check, google-explicit-constructor)

// NOLINTBEGIN(some-other-check, google-explicit-constructor)
// NOLINTEND(some-other-check)
class C5 { C5(int i); };
// NOLINTEND(google-explicit-constructor)

// NOLINTBEGIN(some-other-check, google-explicit-constructor)
// NOLINTEND(google-explicit-constructor)
class C6 { C6(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit
// NOLINTEND(some-other-check)

// NOLINTBEGIN(google-explicit-constructor)
// NOLINTBEGIN(some-other-check)
class C7 { C7(int i); };
// NOLINTEND(some-other-check)
// NOLINTEND(google-explicit-constructor)

// NOLINTBEGIN(google-explicit-constructor)
// NOLINTBEGIN(some-other-check)
class C8 { C8(int i); };
// NOLINTEND(google-explicit-constructor)
// NOLINTEND(some-other-check)

// NOLINTBEGIN(google-explicit-constructor)
// NOLINTBEGIN
class C9 { C9(int i); };
// NOLINTEND
// NOLINTEND(google-explicit-constructor)

// NOLINTBEGIN
// NOLINTBEGIN(google-explicit-constructor)
class C10 { C10(int i); };
// NOLINTEND(google-explicit-constructor)
// NOLINTEND

// NOLINTBEGIN(not-closed-bracket-is-treated-as-skip-all
class C11 { C11(int i); };
// NOLINTEND(not-closed-bracket-is-treated-as-skip-all

// NOLINTBEGIN without-brackets-skip-all, another-check
class C12 { C12(int i); };
// NOLINTEND without-brackets-skip-all, another-check

#define MACRO(X) class X { X(int i); };

MACRO(D1)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit
// CHECK-MESSAGES: :[[@LINE-4]]:28: note: expanded from macro 'MACRO

// NOLINTBEGIN
MACRO(D2)
// NOLINTEND

#define MACRO_NOARG class E { E(int i); };

// NOLINTBEGIN
MACRO_NOARG
// NOLINTEND

// NOLINTBEGIN
#define MACRO_WRAPPED_WITH_NO_LINT class I { I(int i); };
// NOLINTEND

MACRO_WRAPPED_WITH_NO_LINT

#define MACRO_NO_LINT_INSIDE_MACRO \
  /* NOLINTBEGIN */                \
  class J { J(int i); };           \
  /* NOLINTEND */

MACRO_NO_LINT_INSIDE_MACRO

// CHECK-MESSAGES: Suppressed 18 warnings (18 NOLINT).
