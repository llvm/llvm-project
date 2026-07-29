// Test fixture for the vector-like lifetime/role attribute experiment.
//
// Baseline (NO attributes):
//   build-llvm/bin/clang-tidy hicketts/test_hicketts_vector.cpp -- \
//       -std=c++17 -Ihicketts -DHICKETTS_VECTOR_NO_ATTRS
//
// With attributes on:
//   build-llvm/bin/clang-tidy hicketts/test_hicketts_vector.cpp -- \
//       -std=c++17 -Ihicketts

#include "hicketts_vector.h"

using mylib::HickettsVector;

// --- Relational hazard: handles that outlive the container ------------------
// With gsl::Owner/gsl::Pointer + lifetimebound these should warn (-Wdangling).
// Baseline (no attributes) cannot know and stays silent.

int dangling_reference_from_temporary() {
  int &r = HickettsVector<int>{}.front(); // r dangles: temporary destroyed here
  return r;
}

int dangling_iterator_from_temporary() {
  auto it = HickettsVector<int>{}.begin(); // it dangles into destroyed temporary
  return *it;
}

// --- Safe counterparts (should stay silent either way) ----------------------

int safe_reference() {
  HickettsVector<int> v;
  v.push_back(1);
  int &r = v.front(); // v outlives r
  return r;
}

int safe_iterator() {
  HickettsVector<int> v;
  v.push_back(1);
  auto it = v.begin(); // v outlives it
  return *it;
}

// --- Precondition hazard (NOT covered by ANY current attribute) -------------
// front()/pop_back() on an empty vector is UB. Neither the baseline nor the
// Owner/Pointer attributes catch this -- it is the case the PROPOSED
// requires_state("non_empty") role attribute would target. Kept here to show
// the gap the role vocabulary is meant to fill.

int precondition_gap() {
  HickettsVector<int> v;  // empty
  return v.front();       // UB today: no warning from any attribute
}
