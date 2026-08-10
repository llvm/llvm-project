// Test cases for mylib::HickettsOptional — a custom optional-like type
// with differently named functions.
//
// Run from hicketts/ with:
//   ../build-llvm/bin/clang-tidy -checks='bugprone-unchecked-optional-access' \
//     test_hicketts_optional_general.cpp -- -I . -std=c++17 -Wno-undefined-inline

#include "hicketts_optional_general.h"

// --- Unchecked access (should warn if the checker recognises HickettsOptional) ---

static void uncheckedUnwrap(mylib::HickettsOptional<int> &Val) {
  Val.unwrap(); // unchecked access — may be empty
}

static void uncheckedValue(mylib::HickettsOptional<int> &Val) {
  Val.value(); // unchecked access — may be empty
}

static void uncheckedDeref(mylib::HickettsOptional<int> &Val) {
  Val.deref(); // unchecked access — may be empty
}

// --- Checked access (should NOT warn) ---

static void checkedWithBool(mylib::HickettsOptional<int> &Val) {
  if (Val) {
    Val.unwrap(); // safe — checked via operator bool
  }
}

static void checkedValueWithBool(mylib::HickettsOptional<int> &Val) {
  if (Val.has_value()) {
    Val.value(); // safe — checked via operator bool
  }
}

static void checkedWithIsPresent(mylib::HickettsOptional<int> &Val) {
  if (Val.isPresent()) {
    Val.unwrap(); // safe — checked via isPresent()
  }
}

/* static void checkedWithIsEmpty(mylib::HickettsOptional<int> &Val) {
  if (!Val.isEmpty()) {
    Val.unwrap(); // safe — checked via !isEmpty()
  }
} NYI */

// --- State changes ---

// construct() is annotated "emplace(Args&&...)"; the bare "emplace" query matches
// it via the name-part (accept-either) branch -> engaged, so unwrap is safe.
static void safeAfterConstruct(mylib::HickettsOptional<int> &Val) {
  Val.construct(42);
  Val.unwrap(); // safe — just constructed a value
}

static void unsafeAfterClear(mylib::HickettsOptional<int> &Val) {
  Val.construct(42);
  Val.clear();
  Val.unwrap(); // unsafe — value was cleared
}

static void unsafeAfterExchange(mylib::HickettsOptional<int> &A,
                         mylib::HickettsOptional<int> &B) {
  if (A) {
    A.exchange(B);
    A.unwrap(); // unsafe — a's state is now unknown
  }
}

// Works today WITHOUT any annotation: default construction matches no
// constructor case, so has_value is unconstrained -> access conservatively warns.
static void unsafeAfterEmptyConstr() {
  mylib::HickettsOptional<int> A;
  A.unwrap(); // expected: warn (empty)
}

// nothing_t is not std::nullopt_t, so the structural nullopt matcher misses.
// The "optional(std::nullopt_t)" annotation routes this constructor to the
// nullopt transfer (empty) via isOptionalNulloptConstructor's annotation branch,
// so the following unwrap is correctly flagged.
static void unsafeAfterNullConstr() {
  mylib::HickettsOptional<int> A(mylib::nothing);
  A.unwrap(); // warns (empty) — routed to nullopt via the annotation
}

// Works today WITHOUT any annotation: value/conversion constructor case ->
// engaged, so access is safe.
static void safeAfterTypeConstr() {
  mylib::HickettsOptional<int> A(5);
  A.unwrap(); // expected: no warning (engaged)
}

// --- Guarded paths ---

/*static void constructCoversEmptyBranch(mylib::HickettsOptional<int> &Val) {
  if (Val.isEmpty()) {
    Val.construct(99);
  }
  Val.unwrap(); // safe — either was present, or construct filled it
}*/

static void unwrapOrIsAlwaysSafe(mylib::HickettsOptional<int> &Val) {
  int X = Val.unwrapOr(0); // safe — fallback provided
  (void)X;
}

// nothing_t is not std::nullopt_t, so the structural nullopt matcher misses.
// The "operator=(nullopt_t)" annotation routes this assignment to the nullopt
// transfer (empty) — checked before the value/conversion-assignment case — so
// the following unwrap is correctly flagged.
static void unsafeAfterNullAssign() {
  mylib::HickettsOptional<int> A(5);
  A = mylib::nothing;
  A.unwrap(); // warns (empty) — routed to nullopt via the annotation
}
