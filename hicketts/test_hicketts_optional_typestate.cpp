// Test fixture for the hybrid scheme: analyze_as_class + analyze_as_method +
// Clang's existing typestate attributes. Optional only.
//
// Three run modes, each answering a different question:
//
//   1. PARSE CHECK -- do the existing attributes even apply here, on a template,
//      on constructors, on free functions? Expect ZERO diagnostics today.
//        ../build-llvm/bin/clang++ -fsyntax-only -std=c++17 -I . \
//           -Wno-undefined-inline test_hicketts_optional_typestate.cpp
//
//   2. OPTIONAL MODEL -- the POC target. (A) and (B) work today; every (C) case
//      is SILENT until the model learns to read the typestate attributes.
//        ../build-llvm/bin/clang-tidy \
//           -checks='bugprone-unchecked-optional-access' \
//           test_hicketts_optional_typestate.cpp -- -I . -std=c++17 \
//           -Wno-undefined-inline
//
//   3. EXISTING -Wconsumed -- what clang's shipped analysis makes of the SAME
//      annotations. This is the comparison the RFC needs: agreement is the
//      argument for reuse, divergence is the argument against.
//        ../build-llvm/bin/clang++ -fsyntax-only -std=c++17 -I . -Wconsumed \
//           -Wno-undefined-inline test_hicketts_optional_typestate.cpp
//
// Cases have external linkage and are never called, so there is no main() and
// no unused-function noise. Each function is analyzed on its own.

#include "hicketts_optional_typestate.h"

using mylib::HickettsOptional;
using mylib::nothing;

// ===========================================================================
// (A) AUTOMAGIC -- unchanged from the previous fixture, works TODAY.
// Included as the control group: whatever the typestate layer does, these must
// not regress.
// ===========================================================================

void unchecked_value_automagic(HickettsOptional<int> &o) {
  o.value(); // warn (today): unchecked access
}

void checked_with_has_value(HickettsOptional<int> &o) {
  if (o.has_value())
    o.value(); // safe (today)
}

// ===========================================================================
// (B) NAME-MAPPED -- also works TODAY. Control group.
// ===========================================================================

void unchecked_unwrap_namemapped(HickettsOptional<int> &o) {
  o.unwrap(); // warn (today): mapped to value() by name
}

void safe_after_construct(HickettsOptional<int> &o) {
  o.construct(42); // engaged via name-mapped emplace
  o.value();       // safe (today)
}

// ===========================================================================
// (C) TYPESTATE via callable_when -- "this method requires engaged".
// ===========================================================================

// deref() is reachable only through callable_when. Silent today; must warn once
// the model reads the attribute.
void unchecked_deref_callable_when(HickettsOptional<int> &o) {
  o.deref(); // today: silent | target: warn
}

// No-false-positive guard. Narrowed by has_value() (automagic, works today) so
// this case does not also depend on test_typestate landing. Must stay SILENT in
// both modes -- if it warns after the model change, callable_when is being read
// as an unconditional error rather than a precondition.
void checked_deref_callable_when(HickettsOptional<int> &o) {
  if (o.has_value())
    o.deref(); // silent in both modes
}

// ===========================================================================
// (C) TYPESTATE via test_typestate -- "this method queries the state".
// Both polarities come from ONE attribute.
// ===========================================================================

// Positive polarity: isEngaged() must narrow to engaged on the true branch.
void checked_with_is_engaged(HickettsOptional<int> &o) {
  if (o.isEngaged())
    o.value(); // today: warn (false positive) | target: safe
}

// Negative polarity: isEmpty() must narrow to engaged on the FALSE branch.
void checked_with_is_empty(HickettsOptional<int> &o) {
  if (o.isEmpty())
    return;
  o.value(); // today: warn (false positive) | target: safe
}

// Early-return form, negative polarity inverted at the source.
void checked_with_not_is_empty(HickettsOptional<int> &o) {
  if (!o.isEmpty())
    o.value(); // today: warn (false positive) | target: safe
}

// True-positive partner: narrowing must not be unconditional. The EMPTY branch
// of isEngaged() must still warn, or the attribute is being treated as a
// blanket assertion.
void unsafe_on_empty_branch(HickettsOptional<int> &o) {
  if (o.isEngaged())
    return;
  o.value(); // today: warn | target: warn (must NOT go silent)
}

// ===========================================================================
// (C) TYPESTATE via set_typestate -- "this method sets the state".
// ===========================================================================

void safe_after_install(HickettsOptional<int> &o) {
  o.install(42); // set_typestate(unconsumed)
  o.value();     // today: warn (false positive) | target: safe
}

void unsafe_after_clear(HickettsOptional<int> &o) {
  o.install(42);
  o.clear();  // set_typestate(consumed)
  o.value();  // today: warn (for the wrong reason) | target: warn
}

// Assignment, both outcomes, same name -- the overload case name-mapping
// cannot express.
void safe_after_value_assign(HickettsOptional<int> &o) {
  o = 42;    // set_typestate(unconsumed)
  o.value(); // today: warn (false positive) | target: safe
}

void unsafe_after_nothing_assign(HickettsOptional<int> &o) {
  o = 42;
  o = nothing; // set_typestate(consumed)
  o.value();   // today: warn (for the wrong reason) | target: warn
}

// ===========================================================================
// (C) TYPESTATE via return_typestate -- CONSTRUCTORS.
// Same 1-arg shape, opposite outcomes. Name-mapping cannot express this.
// ===========================================================================

void unsafe_nothing_ctor() {
  HickettsOptional<int> o(nothing); // return_typestate(consumed)
  o.value();                        // today: warn | target: warn
}

void safe_value_ctor() {
  HickettsOptional<int> o(42); // return_typestate(unconsumed)
  o.value();                   // today: ? | target: safe
}

// Unannotated default ctor. Consumed.cpp:777 gives default constructors
// CS_Consumed implicitly. Does the new backend inherit that, or does an
// unannotated default ctor leave the state unknown?
void unsafe_default_ctor() {
  HickettsOptional<int> o;
  o.value(); // today: warn | target: warn
}

// ===========================================================================
// FREE FACTORY FUNCTIONS -- return_typestate on a non-member.
// gaps.md:46-47 listed make_optional as a residual hardcoded name with no
// attribute answer. If these two behave, that gap closes.
// ===========================================================================

void safe_from_factory() {
  auto o = mylib::makeEngaged(42); // return_typestate(unconsumed)
  o.value();                       // today: warn (false positive) | target: safe
}

void unsafe_from_empty_factory() {
  auto o = mylib::makeEmpty<int>(); // return_typestate(consumed)
  o.value();                        // today: warn | target: warn
}

// ===========================================================================
// DIVERGENCE PROBE -- where reusing the vocabulary may cost something.
//
// Run mode 3 (-Wconsumed) is the interesting one here. The existing analysis
// hardwires move-source -> consumed (Consumed.cpp:779). For optional that is
// simply false: a moved-from std::optional is still ENGAGED (it holds a
// moved-from T; has_value() stays true). Only .reset() disengages it.
//
// So the same annotation set means two different things to two analyses. If
// -Wconsumed flags the post-move access below and the optional model does not,
// that is a concrete, demonstrable answer to "why not just use -Wconsumed?" --
// and it cuts BOTH ways, so it belongs in the RFC either way.
// ===========================================================================

void moved_from_is_still_engaged(HickettsOptional<int> &o) {
  o.install(42);
  HickettsOptional<int> other(static_cast<HickettsOptional<int> &&>(o));
  (void)other;
  // deref() carries callable_when, so -Wconsumed has a precondition to check
  // here; value() does not, which is why the first draft of this probe was
  // silent under -Wconsumed and proved nothing.
  o.deref();
  o.value(); // optional model: must NOT warn (moved-from optional stays engaged)
}

// Copy leaves both engaged. Both analyses should agree here.
void copy_keeps_state(HickettsOptional<int> &o) {
  o.install(42);
  HickettsOptional<int> other(o);
  other.value(); // target: safe
  o.value();     // target: safe
}

// ===========================================================================
// TEMPLATE INSTANTIATION PROBE.
//
// Consumed.cpp:1199-1206 carries a live FIXME: return_typestate is not properly
// propagated on template instantiation, and the corresponding test in
// SemaDeclAttr.cpp is disabled because of it. HickettsOptional is a template,
// so this is directly in our path. Two instantiations, to see whether the
// attributes survive per-specialization.
// ===========================================================================

void two_instantiations() {
  HickettsOptional<int> i(42);
  HickettsOptional<double> d(3.14);
  i.value(); // target: safe
  d.value(); // target: safe
}
