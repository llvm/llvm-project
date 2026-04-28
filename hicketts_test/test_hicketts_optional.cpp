// Test cases for mylib::HickettsOptional — a custom optional-like type
// with differently named functions.
//
// Run from hicketts_test/ with:
//   clang-tidy -checks='bugprone-unchecked-optional-access' \
//     test_hicketts_optional.cpp -- -I . -Wno-undefined-inline

#include "hicketts_optional.h"

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
  if (Val.hasValue()) {
    Val.value(); // safe — checked via operator bool
  }
}

static void checkedWithIsPresent(mylib::HickettsOptional<int> &Val) {
  if (Val.isPresent()) {
    Val.unwrap(); // safe — checked via isPresent()
  }
}

static void checkedWithIsEmpty(mylib::HickettsOptional<int> &Val) {
  if (!Val.isEmpty()) {
    Val.unwrap(); // safe — checked via !isEmpty()
  }
}

// --- State changes ---

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

// --- Guarded paths ---

static void constructCoversEmptyBranch(mylib::HickettsOptional<int> &Val) {
  if (Val.isEmpty()) {
    Val.construct(99);
  }
  Val.unwrap(); // safe — either was present, or construct filled it
}

static void unwrapOrIsAlwaysSafe(mylib::HickettsOptional<int> &Val) {
  int X = Val.unwrapOr(0); // safe — fallback provided
  (void)X;
}
