//RUN: %clang_analyze_cc1 -std=c++17 -analyze -analyzer-checker=core -verify %s

// expected-no-diagnostics

// Notice the weird alignment.
struct alignas(1024) S {};

void foo() {
  // Operator new() here is the C++17 aligned new that takes two arguments:
  // size and alignment. Size is passed implicitly as usual, and alignment
  // is passed implicitly in a similar manner.
  S *s = new S; // no-warning
  delete s;
}
