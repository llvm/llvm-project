// Test fixture for the HYBRID attribute scheme on mylib::HickettsOptional.
//
// L1 (analyze_as_*) and L3 (gsl::Owner/lifetimebound) are written inline in the
// header; only L2 (proposed roles) is behind -DHO_ROLES. Two run modes:
//
//   default -- L1 + L3. -Wdangling fires via the compiler's lifetime analysis;
//   L1 drives unchecked-optional-access if clang-tidy was built with analyze_as_*:
//     ../build-llvm/bin/clang-tidy \
//        -checks='bugprone-unchecked-optional-access' \
//        test_hicketts_optional_hybrid.cpp -- -I . -std=c++17 -Wno-undefined-inline
//
//   + L2 predicate roles (needs the proposed role attributes implemented):
//     ... -- -I . -std=c++17 -DHO_ROLES -Wno-undefined-inline
//
// Every free function below is analyzed independently of main(); main() only
// exists so the file is a complete program. The unsafe cases are NOT called
// from main (calling them would execute the very UB they document) -- their
// addresses are taken to silence -Wunused-function.

#include "hicketts_optional_hybrid.h"

using mylib::HickettsOptional;
using mylib::nothing;

// === L3: lifetime / dangling (LIVE TODAY via gsl::Owner + lifetimebound) =====

// A handle taken from a temporary optional dangles the moment the full
// expression ends. Expected: -Wdangling.
int dangling_from_temporary() {
  const int &r = HickettsOptional<int>{5}.unwrap(); // temp destroyed here
  return r;
}

// Safe counterpart: the optional outlives the reference. Expected: silent.
int safe_reference() {
  HickettsOptional<int> o{5};
  const int &r = o.unwrap();
  return r;
}

// === L2: unchecked-optional-access predicate (needs -DHO_ROLES) ==============

// Unchecked access -- may be disengaged. Expected: warn.
static void uncheckedUnwrap(HickettsOptional<int> &o) {
  o.unwrap();
}

// Checked via operator bool -> queries_state narrows to engaged. Expected: safe.
static void checkedWithBool(HickettsOptional<int> &o) {
  if (o)
    o.unwrap();
}

// Checked via a NAME-MAPPED query method. Expected: safe.
static void checkedWithIsPresent(HickettsOptional<int> &o) {
  if (o.isPresent())
    o.unwrap();
}

// construct() sets engaged, clear() clears it. Expected: warn after clear.
static void unsafeAfterClear(HickettsOptional<int> &o) {
  o.construct(42);
  o.clear();
  o.unwrap();
}

// Expected: safe -- just constructed a value.
static void safeAfterConstruct(HickettsOptional<int> &o) {
  o.construct(42);
  o.unwrap();
}

// THE disambiguation case, part 1: nullopt-style ctor -> disengaged.
// Expected: warn.
static void unsafeAfterNullCtor() {
  HickettsOptional<int> o(nothing);
  o.unwrap();
}

// THE disambiguation case, part 2: value ctor, SAME 1-arg shape -> engaged.
// Expected: safe. The role on each ctor decl is what tells these two apart --
// no signature string, no std::optional header.
static void safeAfterValueCtor() {
  HickettsOptional<int> o(5);
  o.unwrap();
}

// Nullopt-style assignment -> disengaged. Expected: warn.
static void unsafeAfterNullAssign() {
  HickettsOptional<int> o(5);
  o = nothing;
  o.unwrap();
}

// value_or never accesses an absent value. Expected: safe.
static void unwrapOrIsAlwaysSafe(HickettsOptional<int> &o) {
  int x = o.unwrapOr(0);
  (void)x;
}

int main() {
  HickettsOptional<int> engaged(5);

  // Exercise the safe paths.
  (void)safe_reference();
  checkedWithBool(engaged);
  checkedWithIsPresent(engaged);
  safeAfterConstruct(engaged);
  safeAfterValueCtor();
  unwrapOrIsAlwaysSafe(engaged);

  // Reference the diagnostic-bearing cases without executing their UB, so the
  // analyzer still sees them but the program stays well-defined if run.
  (void)&dangling_from_temporary;
  (void)&uncheckedUnwrap;
  (void)&unsafeAfterClear;
  (void)&unsafeAfterNullCtor;
  (void)&unsafeAfterNullAssign;
  return 0;
}
