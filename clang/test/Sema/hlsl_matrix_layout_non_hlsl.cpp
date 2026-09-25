// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++17 -fsyntax-only -verify -x c++ %s
// expected-no-diagnostics

// row_major and column_major are HLSL-only keywords.
// In non-HLSL modes, they should not be treated as keywords
// and can be used as regular identifiers.
int row_major = 1;
int column_major = 2;

void foo() {
  int row_major = 10;
  int column_major = 20;
}
