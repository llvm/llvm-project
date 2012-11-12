// RUN: %clang_cc1 -verify -fsyntax-only -std=c90 %s
// RUN: %clang_cc1 -verify -fsyntax-only -std=c99 %s
// expected-no-diagnostics

int f (int x)
{
  // sizeof applied to a type should not delete the type.
  return sizeof (int[x]);
}
