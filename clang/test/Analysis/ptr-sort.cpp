// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.nondeterminism.PointerSorting %s -analyzer-output=text -verify

#include "Inputs/system-header-simulator-cxx.h"

bool f (int x) { return true; }
bool g (int *x) { return true; }

void PointerSorting() {
  int a = 1, b = 2, c = 3;
  std::vector<int> V1 = {a, b};
  std::vector<int *> V2 = {&a, &b};

  std::is_sorted(V1.begin(), V1.end());                    // no-warning
  std::nth_element(V1.begin(), V1.begin() + 1, V1.end());  // no-warning
  std::partial_sort(V1.begin(), V1.begin() + 1, V1.end()); // no-warning
  std::sort(V1.begin(), V1.end());                         // no-warning
  std::stable_sort(V1.begin(), V1.end());                  // no-warning
  std::partition(V1.begin(), V1.end(), f);                 // no-warning
  std::stable_partition(V1.begin(), V1.end(), g);          // no-warning

  std::is_sorted(V2.begin(), V2.end()); // expected-warning {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  // expected-note@-1 {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  std::nth_element(V2.begin(), V2.begin() + 1, V2.end()); // expected-warning {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  // expected-note@-1 {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  std::partial_sort(V2.begin(), V2.begin() + 1, V2.end()); // expected-warning {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  // expected-note@-1 {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  std::sort(V2.begin(), V2.end()); // expected-warning {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  // expected-note@-1 {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  std::stable_sort(V2.begin(), V2.end()); // expected-warning {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  // expected-note@-1 {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  std::partition(V2.begin(), V2.end(), f); // expected-warning {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  // expected-note@-1 {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  std::stable_partition(V2.begin(), V2.end(), g); // expected-warning {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
  // expected-note@-1 {{Sorting pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerSorting]
}
