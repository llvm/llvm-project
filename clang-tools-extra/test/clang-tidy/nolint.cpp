// RUN: %check_clang_tidy %s google-explicit-constructor,clang-diagnostic-unused-variable,clang-analyzer-core.UndefinedBinaryOperatorResult %t -- -extra-arg=-Wunused-variable -- -I%S/Inputs/nolint

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

class C1 { C1(int i); }; // NOLINT(*)

class C2 { C2(int i); }; // NOLINT(not-closed-bracket-is-treated-as-skip-all

class C3 { C3(int i); }; // NOLINT(google-explicit-constructor)

class C4 { C4(int i); }; // NOLINT(some-check, google-explicit-constructor)

class C5 { C5(int i); }; // NOLINT without-brackets-skip-all, another-check

void f() {
  int i;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unused variable 'i' [clang-diagnostic-unused-variable]
  int j; // NOLINT
}

#define MACRO(X) class X { X(int i); };
MACRO(D)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit
MACRO(E) // NOLINT

#define MACRO_NOARG class F { F(int i); };
MACRO_NOARG // NOLINT

#define MACRO_NOLINT class G { G(int i); }; // NOLINT
MACRO_NOLINT

#define DOUBLE_MACRO MACRO(H) // NOLINT
DOUBLE_MACRO

// CHECK-MESSAGES: Suppressed 12 warnings (12 NOLINT)
