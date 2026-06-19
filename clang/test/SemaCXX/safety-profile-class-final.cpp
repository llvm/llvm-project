// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -fprofiles-test-profiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::class_final)]];
// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::other)]];

struct Plain { // expected-error {{test profile fired on completion of class 'Plain' under profile 'test::class_final'}}
  int m;
};

class PlainClass { // expected-error {{test profile fired on completion of class 'PlainClass' under profile 'test::class_final'}}
public:
  int m;
};

union PlainUnion { // expected-error {{test profile fired on completion of class 'PlainUnion' under profile 'test::class_final'}}
  int a;
  float b;
};

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::class_final)]] SuppressedOnClass {
  int m;
};

// Suppress with rule restriction (the implicit-rule profile uses rule "").
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::class_final, rule: "")]] SuppressedOnClassByRule {
  int m;
};

// Non-matching profile does not suppress.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::other)]] NotSuppressedWrongProfile { // expected-error {{test profile fired on completion of class 'NotSuppressedWrongProfile' under profile 'test::class_final'}}
  int m;
};

// Lambdas are filtered out by the dispatcher: the implicit closure types
// are not diagnosed even though they are CXXRecordDecls that go through
// CheckCompletedCXXClass.
void test_lambda_skip() {
  auto f = []() { return 1; };
  auto g = [](int x) { return x; };
  (void)f; (void)g;
}

// Generic lambdas: the closure's call operator is a template, but the
// closure type itself is still a lambda and must be filtered.
void test_generic_lambda_skip() {
  auto f = [](auto x) { return x; };
  (void)f(0);
  (void)f(0.0);
}

// A dependent primary template does not fire on the template itself; the
// instantiated specialization's diagnostic is emitted at the primary
// template's location with a note at the instantiation site.
template <typename T>
struct PrimaryTemplate { // expected-error {{test profile fired on completion of class 'PrimaryTemplate<int>' under profile 'test::class_final'}} \
                         // expected-error {{test profile fired on completion of class 'PrimaryTemplate<float>' under profile 'test::class_final'}}
  T m;
};

PrimaryTemplate<int> instantiate_primary; // expected-note {{in instantiation of template class 'PrimaryTemplate<int>' requested here}}

PrimaryTemplate<float> instantiate_primary_float; // expected-note {{in instantiation of template class 'PrimaryTemplate<float>' requested here}}

// Explicit specialization is a fresh definition and fires at its own line.
template <>
struct PrimaryTemplate<char> { // expected-error {{test profile fired on completion of class 'PrimaryTemplate<char>' under profile 'test::class_final'}}
  char m;
};

// Suppress on the primary template carries through instantiation via the
// lexical-parent walk on the instantiated CXXRecordDecl.
template <typename T>
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::class_final)]] SuppressedTemplate {
  T m;
};
SuppressedTemplate<int> suppressed_template_inst;

// Suppress on an enclosing namespace silences a nested class via the
// lexical-parent walk.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
namespace [[profiles::suppress(test::class_final)]] suppressed_ns {
  struct NestedInSuppressedNS {
    int m;
  };
  struct AlsoNested {
    int m;
  };
}

// Suppress on an enclosing class silences a nested class.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::class_final)]] OuterSuppressed { // OuterSuppressed itself is suppressed.
  struct Inner { // Inner reaches OuterSuppressed via lexical-parent walk.
    int m;
  };
  struct AlsoInner {
    int m;
  };
};

// Suppress on an enclosing namespace also silences template instantiations
// because the instantiated CXXRecordDecl's lexical parent chain still
// reaches the namespace.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
namespace [[profiles::suppress(test::class_final)]] suppressed_template_ns {
  template <typename T>
  struct InsideSuppressedNS {
    T m;
  };
}
suppressed_template_ns::InsideSuppressedNS<int> inside_suppressed_ns_inst;

// Non-matching suppress at namespace scope does not silence the inner class.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
namespace [[profiles::suppress(test::other)]] wrong_profile_ns {
  struct NotSuppressed { // expected-error {{test profile fired on completion of class 'NotSuppressed' under profile 'test::class_final'}}
    int m;
  };
}

// SFINAE: a class template whose instantiation would fire the profile
// diagnostic must not cause the substitution to fail. ProfileRuleError uses
// SFINAE_Suppress, so the diagnostic is suppressed during deduction; the
// first overload is selected because the substitution succeeds. The
// diagnostic is then replayed at the class definition (with the usual
// instantiation-context notes).
template <typename T>
struct SfinaeTriggered { // expected-error {{test profile fired on completion of class 'SfinaeTriggered<long>' under profile 'test::class_final'}}
  using type = T;
};

template <typename T>
auto sfinae_pick(T) -> typename SfinaeTriggered<T>::type { return T{}; } // expected-note {{in instantiation of template class 'SfinaeTriggered<long>' requested here}}

template <typename T>
auto sfinae_pick(...) -> int { return 1; }

static_assert(__is_same(decltype(sfinae_pick<long>(0L)), long), // expected-note {{while substituting explicitly-specified template arguments into function template 'sfinae_pick'}}
              "profile violation must not SFINAE out the first overload");

// Local classes inside a function body fire (a function is not itself a
// class-finalization subject).
void test_local_class() {
  struct LocalInFn { int m; }; // expected-error {{test profile fired on completion of class 'LocalInFn' under profile 'test::class_final'}}
  LocalInFn x;
  (void)x;
}

// Function-level suppress silences a local class defined in its body via
// the parse-time ProfileSuppressStack (not via the lexical-parent walk;
// the walk goes through the *Decl* chain, not statement context).
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::class_final)]]
void test_local_class_suppressed_via_fn() {
  struct LocalInSuppressedFn { int m; };
  LocalInSuppressedFn x;
  (void)x;
}

// Local classes inside a lambda *body* still fire. Only the closure type
// itself is filtered by isLambda(); user-defined classes nested inside the
// closure's call operator are not lambdas.
void test_local_class_inside_lambda() {
  auto f = []() {
    struct LocalInLambda { int m; }; // expected-error {{test profile fired on completion of class 'LocalInLambda' under profile 'test::class_final'}}
    LocalInLambda x;
    (void)x;
  };
  f();
}

// Anonymous union and anonymous struct members fire with synthesized
// "(unnamed ...)" diagnostic names.
struct HasAnonymousMembers { // expected-error {{test profile fired on completion of class 'HasAnonymousMembers' under profile 'test::class_final'}}
  union { // expected-error {{test profile fired on completion of class '(unnamed union}}
    int a;
    float b;
  };
  struct { // expected-error {{test profile fired on completion of class '(unnamed struct}}
    int x;
  };
};

// Class template partial specialization instantiation. The partial
// specialization itself is dependent (skipped); its concrete instantiation
// fires at the primary template's location with a note at the use site.
template <typename T> struct PartialSpec { T m; }; // expected-error {{test profile fired on completion of class 'PartialSpec<int *>' under profile 'test::class_final'}}
template <typename T> struct PartialSpec<T*> {
  T *p;
};
void use_partial_spec() {
  PartialSpec<int *> x; // expected-note {{in instantiation of template class 'PartialSpec<int *>' requested here}}
  (void)x;
}

// Explicit instantiation *definition* fires at the explicit-instantiation
// directive's own line (unlike implicit instantiation, which fires at the
// primary template's line).
template <typename T> struct ExplicitInst { T m; };
template struct ExplicitInst<int>; // expected-error {{test profile fired on completion of class 'ExplicitInst<int>' under profile 'test::class_final'}} \
                                   // expected-note {{in instantiation of template class 'ExplicitInst<int>' requested here}}

// Explicit instantiation *declaration* (extern template) still instantiates
// the class itself per C++ rules, so it also fires at its own line.
extern template struct ExplicitInst<short>; // expected-error {{test profile fired on completion of class 'ExplicitInst<short>' under profile 'test::class_final'}} \
                                            // expected-note {{in instantiation of template class 'ExplicitInst<short>' requested here}}

// Friend class definitions are completed normally and fire at their own line.
class HasFriendDecl { // expected-error {{test profile fired on completion of class 'HasFriendDecl' under profile 'test::class_final'}}
  friend struct FriendedLater;
};
struct FriendedLater { int m; }; // expected-error {{test profile fired on completion of class 'FriendedLater' under profile 'test::class_final'}}

// Without `-fprofiles`, the enforce attribute is `warn_attribute_ignored`
// and the diagnostic never fires. This is exercised by the no-profiles RUN
// line, which expects only the two attribute-ignored warnings above.
