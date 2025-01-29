// REQUIRES: static-analyzer
// RUN: %check_clang_tidy %s google-explicit-constructor,clang-diagnostic-unused-variable,clang-analyzer-core.UndefinedBinaryOperatorResult,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-pro-type-member-init %t -- -extra-arg=-Wunused-variable -- -I%S/Inputs/nolint

#include "trigger_warning.h"
void I(int& Out) {
  int In;
  A1(In, Out);
}
// CHECK-MESSAGES-NOT: trigger_warning.h:{{.*}} warning
// CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}} note

class A { A(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

class B { B(int i); }; // NOLINT

class C { C(int i); }; // NOLINT(for-some-other-check)
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

class C1 { C1(int i); }; // NOLINT()
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

class C2 { C2(int i); }; // NOLINT(*)

class C3 { C3(int i); }; // NOLINT(not-closed-bracket-is-treated-as-skip-all

class C4 { C4(int i); }; // NOLINT(google-explicit-constructor)

class C5 { C5(int i); }; // NOLINT(some-check, google-explicit-constructor)

class C6 { C6(int i); }; // NOLINT without-brackets-skip-all

// Other NOLINT* types (e.g. NEXTLINE) should not be misconstrued as a NOLINT:
class C7 { C7(int i); }; // NOLINTNEXTLINE
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

// NOLINT must be UPPERCASE:
// NOLINTnextline
class C8 { C8(int i); }; // nolint
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

// Unrecognized marker:
// NOLINTNEXTLINEXYZ
class C9 { C9(int i); }; // NOLINTXYZ
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

// C-style comments are supported:
class C10 { C10(int i); }; /* NOLINT */
/* NOLINT */ class C11 { C11(int i); };

// Multiple NOLINTs in the same comment:
class C12 { C12(int i); }; // NOLINT(some-other-check) NOLINT(google-explicit-constructor)
class C13 { C13(int i); }; // NOLINT(google-explicit-constructor) NOLINT(some-other-check)
class C14 { C14(int i); }; // NOLINTNEXTLINE(some-other-check) NOLINT(google-explicit-constructor)

// NOLINTNEXTLINE(google-explicit-constructor) NOLINT(some-other-check)
class C15 { C15(int i); }; 

// Any text after a NOLINT expression is treated as a comment:
class C16 { C16(int i); }; // NOLINT: suppress check because <reason>
class C17 { C17(int i); }; // NOLINT(google-explicit-constructor): suppress check because <reason>

// NOLINT must appear in its entirety on one line:
class C18 { C18(int i); }; /* NOL
INT */
// CHECK-MESSAGES: :[[@LINE-2]]:13: warning: single-argument constructors must be marked explicit

/* NO
LINT */ class C19 { C19(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: single-argument constructors must be marked explicit

// Spaces between items in the comma-separated check list are ignroed:
class C20 { C20(int i); }; // NOLINT( google-explicit-constructor )
class C21 { C21(int i); }; // NOLINT( google-explicit-constructor , some-other-check )
class C22 { C22(int i); }; // NOLINT(google-explicit- constructor)
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: single-argument constructors must be marked explicit

// If there is a space between "NOLINT" and the bracket, it is treated as a regular NOLINT: 
class C23 { C23(int i); }; // NOLINT (some-other-check)

void f() {
  int i;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unused variable 'i' [clang-diagnostic-unused-variable]
  int j; // NOLINT
  int k; // NOLINT(clang-diagnostic-unused-variable)
}

#define MACRO(X) class X { X(int i); };
MACRO(D)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit
MACRO(E) // NOLINT

#define MACRO_NOARG class F { F(int i); };
MACRO_NOARG // NOLINT

#define MACRO_NOLINT class G { G(int i); }; // NOLINT
MACRO_NOLINT

// Check that we can suppress diagnostics about macro arguments (as opposed to
// diagnostics about the macro contents itself).
#define MACRO_SUPPRESS_DIAG_FOR_ARG_1(X)                \
  class X {                                             \
    X(int i); /* NOLINT(google-explicit-constructor) */ \
  };

MACRO_SUPPRESS_DIAG_FOR_ARG_1(G1)

#define MACRO_SUPPRESS_DIAG_FOR_ARG_2(X)                          \
  struct X { /* NOLINT(cppcoreguidelines-pro-type-member-init) */ \
    int a = 0;                                                    \
    int b;                                                        \
  };

MACRO_SUPPRESS_DIAG_FOR_ARG_2(G2)

#define DOUBLE_MACRO MACRO(H) // NOLINT
DOUBLE_MACRO

class D1 { D1(int x); }; // NOLINT(google*)
class D2 { D2(int x); }; // NOLINT(*explicit-constructor)
class D3 { D3(int x); }; // NOLINT(*explicit*)
class D4 { D4(int x); }; // NOLINT(-explicit-constructor)
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit
class D5 { D5(int x); }; // NOLINT(google*,-google*)
class D6 { D6(int x); }; // NOLINT(*,-google*)

int array1[10];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead [cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays]

int array2[10];  // NOLINT(cppcoreguidelines-avoid-c-arrays)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

int array3[10];  // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
int array4[10];  // NOLINT(*-avoid-c-arrays)

// CHECK-MESSAGES: Suppressed 36 warnings (36 NOLINT)
