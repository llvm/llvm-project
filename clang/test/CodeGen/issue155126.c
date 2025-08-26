// RUN: %clang_cc1 -std=c23 -verify %s
// RUN: %clang_cc1 -std=c23 -verify -fexperimental-new-constant-interpreter %s

// expected-no-diagnostics

enum e : bool { b = true };
void foo ()
{
  enum e e1;
  e1 = (bool) nullptr;
}
