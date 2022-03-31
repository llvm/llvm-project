// Check whether output generation options (like -save-temps) will not affect
// the execution of the analyzer.

// RUN: clang-check -analyze %s -- -save-temps -c -Xclang -verify

// Check whether redundant -fsyntax-only options will affect the execution of
// the analyzer.

// RUN: clang-check -analyze %s -- \
// RUN:   -fsyntax-only -c -fsyntax-only -Xclang -verify 2>&1 | \
// RUN:   FileCheck %s --allow-empty

// CHECK-NOT: argument unused during compilation: '--analyze'

void a(int *x) {
  if (x) {
  }
  *x = 47; // expected-warning {{Dereference of null pointer}}
}
