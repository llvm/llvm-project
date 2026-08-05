// Test fixture for the HYBRID attribute scheme on mylib::HickettsOptional.
//
// Two run modes:
//   default -- (A) automagic + (B) name-mapped analysis works; (C) role-only
//   cases are silent (roles not implemented yet). Plus -Wdangling (L3).
//     ../build-llvm/bin/clang-tidy \
//        -checks='bugprone-unchecked-optional-access' \
//        test_hicketts_optional_hybrid.cpp -- -I . -std=c++17 -Wno-undefined-inline
//
//   -DHO_ROLES -- additionally enables the (C) behavioural cases (once the role
//   attributes are implemented in the model).
//
// Free functions are analyzed independently of main(); the unsafe cases are not
// called (their addresses are taken to silence -Wunused-function) so the program
// stays well-defined if run.

#include "hicketts_optional_hybrid.h"

using mylib::HickettsOptional;
using mylib::nothing;

// === L3 lifetime: live today via gsl::Owner + lifetimebound =================
int dangling_from_temporary() {
  const int &r = HickettsOptional<int>{5}.value(); // -Wdangling: temp dies here
  return r;
}

// === (A) AUTOMAGIC: value()/has_value() recognised via analyze_as_class =====
// Works TODAY -- structural recognition, no role needed.
static void unchecked_value_automagic(HickettsOptional<int> &o) {
  o.value(); // warn (today): unchecked access, recognised structurally
}
static void checked_with_has_value(HickettsOptional<int> &o) {
  if (o.has_value())
    o.value(); // safe (today)
}

// === (B) NAME-MAPPED: unwrap()->value, isPresent()->has_value, etc. =========
// Works TODAY -- name-only analyze_as_method.
static void unchecked_unwrap_namemapped(HickettsOptional<int> &o) {
  o.unwrap(); // warn (today): mapped to value() by name
}
static void checked_with_is_present(HickettsOptional<int> &o) {
  if (o.isPresent())
    o.unwrap(); // safe (today)
}
static void safe_after_construct(HickettsOptional<int> &o) {
  o.construct(42); // engaged via name-mapped emplace
  o.value();       // safe (today)
}
static void unsafe_after_clear(HickettsOptional<int> &o) {
  o.construct(42);
  o.clear();  // disengaged via name-mapped reset
  o.value();  // warn (today)
}

// === (C) BEHAVIOURAL: role-only -- SILENT today, correct under -DHO_ROLES ====

// deref() is role-only (assume_engaged): unrecognised today, so no warning;
// warns once roles are implemented.
static void unchecked_deref_role(HickettsOptional<int> &o) {
  o.deref(); // today: silent; -DHO_ROLES: warn
}

// THE test_engaged trial (positive-polarity query, role-only). Unlike the other
// (C) cases -- which are false NEGATIVES today -- this is a false POSITIVE: the
// guard isn't recognised, so value() warns even though it is actually safe. When
// test_engaged is implemented, THIS LINE should flip to silent under -DHO_ROLES.
static void checked_with_role_query(HickettsOptional<int> &o) {
  if (o.isEngaged())
    o.value(); // today: WARN (test_engaged not implemented); -DHO_ROLES: safe
}

// Constructor disambiguation -- the case roles exist to FIX. Today the nullopt
// ctor falls through to structural value-construction and is wrongly treated as
// engaged (a false negative); the disengaged role corrects it.
static void unsafe_after_null_ctor() {
  HickettsOptional<int> o(nothing); // today: wrongly engaged; role: disengaged
  o.value();                        // today: silent (FALSE NEGATIVE); -DHO_ROLES: warn
}
static void safe_after_value_ctor() {
  HickettsOptional<int> o(5); // engaged (structural value-ctor, and role agrees)
  o.value();                  // safe (today and with roles)
}

// Assignment disambiguation -- same story as the ctors.
static void unsafe_after_null_assign() {
  HickettsOptional<int> o(5);
  o = nothing; // today: wrongly engaged (value-assign); role: disengaged
  o.value();   // today: silent (FALSE NEGATIVE); -DHO_ROLES: warn
}

// value_or never accesses an absent value.
static void unwrap_or_is_always_safe(HickettsOptional<int> &o) {
  int x = o.unwrapOr(0);
  (void)x;
}

int main() {
  HickettsOptional<int> engaged(5);

  // Safe paths.
  checked_with_has_value(engaged);
  checked_with_is_present(engaged);
  safe_after_construct(engaged);
  safe_after_value_ctor();
  unwrap_or_is_always_safe(engaged);

  // Diagnostic-bearing cases: referenced, not executed.
  (void)&dangling_from_temporary;
  (void)&unchecked_value_automagic;
  (void)&unchecked_unwrap_namemapped;
  (void)&unsafe_after_clear;
  (void)&unchecked_deref_role;
  (void)&checked_with_role_query;
  (void)&unsafe_after_null_ctor;
  (void)&unsafe_after_null_assign;
  return 0;
}
