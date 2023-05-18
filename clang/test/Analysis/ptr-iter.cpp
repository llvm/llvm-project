// RUN: %clang_analyze_cc1 %s -std=c++14 -analyzer-output=text -verify \
// RUN: -analyzer-checker=core,alpha.nondeterminism.PointerIteration

#include "Inputs/system-header-simulator-cxx.h"

template<class T>
void f(T x);

void PointerIteration() {
  int a = 1, b = 2;
  std::set<int> OrderedIntSet = {a, b};
  std::set<int *> OrderedPtrSet = {&a, &b};
  std::unordered_set<int> UnorderedIntSet = {a, b};
  std::unordered_set<int *> UnorderedPtrSet = {&a, &b};

  for (auto i : OrderedIntSet) // no-warning
    f(i);

  for (auto i : OrderedPtrSet) // no-warning
    f(i);

  for (auto i : UnorderedIntSet) // no-warning
    f(i);

  for (auto i : UnorderedPtrSet) // expected-warning {{Iteration of pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerIteration]
// expected-note@-1 {{Iteration of pointer-like elements can result in non-deterministic ordering}} [alpha.nondeterminism.PointerIteration]
    f(i);
}
