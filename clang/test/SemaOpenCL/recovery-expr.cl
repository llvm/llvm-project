// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++ -frecovery-ast

void kernel nocrash() {
  constant int L1 = 0;

  private int *constant L2 = L1++; // expected-error {{read-only variable is not assignable}}
}
