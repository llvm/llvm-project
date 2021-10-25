// REQUIRES: static-analyzer
// RUN: %check_clang_tidy %s google-explicit-constructor,clang-diagnostic-unused-variable,clang-analyzer-core.UndefinedBinaryOperatorResult,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays %t -- -extra-arg=-Wunused-variable -- -I%S/Inputs/nolint

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

class C6 { C6(int i); }; // NOLINT without-brackets-skip-all, another-check

class C7 { C7(int i); }; // NOLINTNEXTLINE doesn't get misconstrued as a NOLINT
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

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
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead [cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays]

int array2[10];  // NOLINT(cppcoreguidelines-avoid-c-arrays)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead [modernize-avoid-c-arrays]

int array3[10];  // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
int array4[10];  // NOLINT(*-avoid-c-arrays)

// CHECK-MESSAGES: Suppressed 23 warnings (23 NOLINT)
