// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyze-function=testVariablyModifiedTypes -verify %s

// Test that we process variably modified type correctly - the call graph construction should pick up function_with_bug while recursively visiting test_variably_modifiable_types.
unsigned getArraySize(int *x) {
  if (!x)
    return *x; // expected-warning {{Dereference of null pointer}}
  return 1;
}

int testVariablyModifiedTypes(int *x) {
  int mytype[getArraySize(x)];
  return 0;
}
