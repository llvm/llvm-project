// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s \
// RUN:   -analyze-function="Window::overloaded(int)"

// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s \
// RUN:   -analyze-function="c:@S@Window@F@overloaded#I#"

// RUN: %clang_extdef_map %s | FileCheck %s
// CHECK:      27:c:@S@Window@F@overloaded#I#
// CHECK-NEXT: 27:c:@S@Window@F@overloaded#C#
// CHECK-NEXT: 27:c:@S@Window@F@overloaded#d#

void clang_analyzer_warnIfReached();

struct Window {
  void overloaded(double) { clang_analyzer_warnIfReached(); } // not analyzed, thus not reachable
  void overloaded(char) { clang_analyzer_warnIfReached(); }   // not analyzed, thus not reachable
  void overloaded(int) { clang_analyzer_warnIfReached(); } // expected-warning {{REACHABLE}}
};
