// RUN: %clang_analyze_cc1 -verify %s -fcxx-exceptions -fexceptions -analyzer-checker=core,alpha.deadcode.UnreachableCode

// expected-no-diagnostics

void foo();

void f4() {
  try {
    foo();
  } catch (int) {
  }
}