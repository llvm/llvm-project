// Test cases for mylib::HickettsOptional — a custom optional-like type
// with differently named functions.
//
// Run from hicketts_test/ with:
//   clang-tidy -checks='bugprone-unchecked-optional-access' \
//     test_hicketts_optional.cpp -- -I . -Wno-undefined-inline

#include "hicketts_optional.h"

// --- Unchecked access (should warn if the checker recognises HickettsOptional) ---

void unchecked_unwrap(mylib::HickettsOptional<int> &val) {
  val.unwrap(); // unchecked access — may be empty
}

void unchecked_deref(mylib::HickettsOptional<int> &val) {
  val.deref(); // unchecked access — may be empty
}

// --- Checked access (should NOT warn) ---

void checked_with_bool(mylib::HickettsOptional<int> &val) {
  if (val) {
    val.unwrap(); // safe — checked via operator bool
  }
}

void checked_with_isPresent(mylib::HickettsOptional<int> &val) {
  if (val.isPresent()) {
    val.unwrap(); // safe — checked via isPresent()
  }
}

void checked_with_isEmpty(mylib::HickettsOptional<int> &val) {
  if (!val.isEmpty()) {
    val.unwrap(); // safe — checked via !isEmpty()
  }
}

// --- State changes ---

void safe_after_construct(mylib::HickettsOptional<int> &val) {
  val.construct(42);
  val.unwrap(); // safe — just constructed a value
}

void unsafe_after_clear(mylib::HickettsOptional<int> &val) {
  val.construct(42);
  val.clear();
  val.unwrap(); // unsafe — value was cleared
}

void unsafe_after_exchange(mylib::HickettsOptional<int> &a,
                           mylib::HickettsOptional<int> &b) {
  if (a) {
    a.exchange(b);
    a.unwrap(); // unsafe — a's state is now unknown
  }
}

// --- Guarded paths ---

void construct_covers_empty_branch(mylib::HickettsOptional<int> &val) {
  if (val.isEmpty()) {
    val.construct(99);
  }
  val.unwrap(); // safe — either was present, or construct filled it
}

void unwrapOr_is_always_safe(mylib::HickettsOptional<int> &val) {
  int x = val.unwrapOr(0); // safe — fallback provided
  (void)x;
}
