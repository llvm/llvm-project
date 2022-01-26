// RUN: %check_clang_tidy %s google-explicit-constructor,clang-diagnostic-unused-variable,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays %t -- -extra-arg=-Wunused-variable

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

// NOLINTBEGIN()
class C2 { C2(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit
// NOLINTEND()

// NOLINTBEGIN(*)
class C3 { C3(int i); };
// NOLINTEND(*)

// NOLINTBEGIN(some-other-check)
class C4 { C4(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit
// NOLINTEND(some-other-check)

// NOLINTBEGIN(some-other-check, google-explicit-constructor)
class C5 { C5(int i); };
// NOLINTEND(some-other-check, google-explicit-constructor)

// NOLINTBEGIN(google-explicit-constructor)
// NOLINTBEGIN(some-other-check)
class C6 { C6(int i); };
// NOLINTEND(some-other-check)
// NOLINTEND(google-explicit-constructor)

// NOLINTBEGIN(google-explicit-constructor)
// NOLINTBEGIN(some-other-check)
class C7 { C7(int i); };
// NOLINTEND(google-explicit-constructor)
// NOLINTEND(some-other-check)

// NOLINTBEGIN(google-explicit-constructor)
// NOLINTBEGIN
class C8 { C8(int i); };
// NOLINTEND
// NOLINTEND(google-explicit-constructor)

// NOLINTBEGIN
// NOLINTBEGIN(google-explicit-constructor)
class C9 { C9(int i); };
// NOLINTEND(google-explicit-constructor)
// NOLINTEND

// NOLINTBEGIN(not-closed-bracket-is-treated-as-skip-all
class C10 { C10(int i); };
// NOLINTEND(not-closed-bracket-is-treated-as-skip-all

// NOLINTBEGIN without-brackets-skip-all, another-check
class C11 { C11(int i); };
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

// NOLINTBEGIN(google*)
class C12 { C12(int i); };
// NOLINTEND(google*)

// NOLINTBEGIN(*explicit-constructor)
class C15 { C15(int i); };
// NOLINTEND(*explicit-constructor)

// NOLINTBEGIN(*explicit*)
class C16 { C16(int i); };
// NOLINTEND(*explicit*)

// NOLINTBEGIN(-explicit-constructor)
class C17 { C17(int x); };
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: single-argument constructors must be marked explicit
// NOLINTEND(-explicit-constructor)

// NOLINTBEGIN(google*,-google*)
class C18 { C18(int x); };
// NOLINTEND(google*,-google*)

// NOLINTBEGIN(*,-google*)
class C19 { C19(int x); };
// NOLINTEND(*,-google*)

int array1[10];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead [cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays]

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
int array2[10];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead [modernize-avoid-c-arrays]
// NOLINTEND(cppcoreguidelines-avoid-c-arrays)

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
int array3[10];
// NOLINTEND(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)

// NOLINTBEGIN(*-avoid-c-arrays)
int array4[10];
// NOLINTEND(*-avoid-c-arrays)

// CHECK-MESSAGES: Suppressed 27 warnings (27 NOLINT).
