// Test fixture for the vector-like lifetime + predicate-role experiment.
//
// Real attributes are inline and always on. Default run:
//   build-llvm/bin/clang-tidy hicketts/test_hicketts_vector_hybrid.cpp -- \
//       -std=c++17 -Ihicketts
//
// + proposed predicate roles (needs the role attributes implemented):
//   build-llvm/bin/clang-tidy hicketts/test_hicketts_vector_hybrid.cpp -- \
//       -std=c++17 -Ihicketts -DHV_ROLES

#include "hicketts_vector_hybrid.h"

using mylib::HickettsVector;

// --- L3 relational hazard: handles that outlive the container ---------------
// gsl::Owner/Pointer + lifetimebound -> -Wdangling. Baseline stays silent.

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

// --- L2 predicate precondition (needs -DHV_ROLES) ---------------------------
// front()/pop_back() on an empty vector is UB. This is the assume_engaged role,
// the direct analog of optional's value() requiring engaged. Neither the
// baseline nor Owner/Pointer catches it -- the gap the roles are meant to fill.

int precondition_gap() {
  HickettsVector<int> v; // empty (disengaged)
  return v.front();      // warn under -DHV_ROLES: assume_engaged not established
}

int safe_after_push() {
  HickettsVector<int> v;
  v.push_back(1);   // engaged
  return v.front(); // safe
}

int unsafe_after_clear() {
  HickettsVector<int> v;
  v.push_back(1);
  v.clear();        // disengaged
  return v.front(); // warn under -DHV_ROLES
}

// --- L2 polarity: empty() is a NEGATIVE-polarity test -----------------------
// The optional fixture narrows via has_value() (true == engaged); the vector
// narrows via empty() (true == disengaged). Same bit, opposite polarity -- both
// must make the guarded front() safe. This is the case optional cannot show.

int safe_guarded_by_not_empty(HickettsVector<int> &v) {
  if (!v.empty())
    return v.front(); // safe: !empty() -> engaged
  return 0;
}

int safe_guarded_by_empty_early_return(HickettsVector<int> &v) {
  if (v.empty())
    return 0;
  return v.front(); // safe: fallthrough -> engaged
}
