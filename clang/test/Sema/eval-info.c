// RUN: %clang_cc1 %s -fsyntax-only -triple x86_64-unknown-windows-msvc -verify

// expected-no-diagnostics

// Make sure the new constant interpolator is not enabled unintentionally
// to cause assertion.
typedef enum x {
  a = 1,
} x;
