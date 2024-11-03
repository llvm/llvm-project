// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux -analyzer-checker=alpha.unix.StdCLibraryFunctions,debug.ExprInspection -verify %s

// Test that we don't model functions with broken prototypes.
// Because they probably work differently as well.
//
// This test lives in a separate file because we wanted to test all functions
// in the .c file, however in C there are no overloads.

void clang_analyzer_eval(bool);
bool isalpha(char);

void test() {
  clang_analyzer_eval(isalpha('A')); // no-crash // expected-warning{{UNKNOWN}}
}
