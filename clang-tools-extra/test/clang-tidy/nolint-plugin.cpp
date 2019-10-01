// REQUIRES: static-analyzer
// RUN: c-index-test -test-load-source-reparse 2 all %s -Xclang -add-plugin -Xclang clang-tidy -Xclang -plugin-arg-clang-tidy -Xclang -checks='-*,google-explicit-constructor,clang-diagnostic-unused-variable,clang-analyzer-core.UndefinedBinaryOperatorResult' -Wunused-variable -I%S/Inputs/nolint 2>&1 | FileCheck %s

#include "trigger_warning.h"
void I(int& Out) {
  int In;
  A1(In, Out);
}
// CHECK-NOT: trigger_warning.h:{{.*}} warning
// CHECK-NOT: :[[@LINE-4]]:{{.*}} note

class A { A(int i); };
// CHECK-DAG: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

class B { B(int i); }; // NOLINT

class C { C(int i); }; // NOLINT(for-some-other-check)
// CHECK-DAG: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

class C1 { C1(int i); }; // NOLINT(*)

class C2 { C2(int i); }; // NOLINT(not-closed-bracket-is-treated-as-skip-all

class C3 { C3(int i); }; // NOLINT(google-explicit-constructor)

class C4 { C4(int i); }; // NOLINT(some-check, google-explicit-constructor)

class C5 { C5(int i); }; // NOLINT without-brackets-skip-all, another-check

void f() {
  int i;
// CHECK-DAG: :[[@LINE-1]]:7: warning: unused variable 'i' [-Wunused-variable]
//                          31:7: warning: unused variable 'i' [-Wunused-variable]
//  int j; // NOLINT
//  int k; // NOLINT(clang-diagnostic-unused-variable)
}

#define MACRO(X) class X { X(int i); };
MACRO(D)
// CHECK-DAG: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit
MACRO(E) // NOLINT

#define MACRO_NOARG class F { F(int i); };
MACRO_NOARG // NOLINT

#define MACRO_NOLINT class G { G(int i); }; // NOLINT
MACRO_NOLINT

#define DOUBLE_MACRO MACRO(H) // NOLINT
DOUBLE_MACRO
