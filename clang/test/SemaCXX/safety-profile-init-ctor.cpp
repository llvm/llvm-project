// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

struct WithCtor { WithCtor(); };
struct Inner { int y; };
struct InnerMarked { int y [[uninitialized]]; };
struct InnerMixed { int a [[uninitialized]]; int b; };

struct MissingMember {
  int x; // expected-note {{member 'x' declared here}}
  MissingMember() {} // expected-error {{constructor does not initialize member 'x' under profile 'std::init'}}
};

struct MemInit {
  int x;
  MemInit() : x(0) {}
};

struct DefaultMemberInit {
  int x = 0;
  DefaultMemberInit() {}
};

struct Marked {
  int x [[uninitialized]];
  Marked() {}
};

struct BodyAssignment {
  int x; // expected-note {{member 'x' declared here}}
  // A plain body assignment is not initialization for this rule.
  BodyAssignment() { x = 0; } // expected-error {{constructor does not initialize member 'x' under profile 'std::init'}}
};

struct NestedAggregate {
  Inner m; // expected-note {{member 'm' declared here}}
  NestedAggregate() {} // expected-error {{constructor does not initialize member 'm' under profile 'std::init'}}
};

// A member whose type's only indeterminate scalar is [[uninitialized]] is
// acknowledged (paper §6.2), so the constructor need not initialize it.
struct NestedMarkedMember {
  InnerMarked m;
  NestedMarkedMember() {}
};

// A member whose type still leaves an unacknowledged scalar indeterminate fires.
struct NestedMixedMember {
  InnerMixed m; // expected-note {{member 'm' declared here}}
  NestedMixedMember() {} // expected-error {{constructor does not initialize member 'm' under profile 'std::init'}}
};

struct TrustedMemberCtor {
  WithCtor m;
  TrustedMemberCtor() {}
};

struct PartialInit {
  int x;
  int y; // expected-note {{member 'y' declared here}}
  PartialInit() : x(0) {} // expected-error {{constructor does not initialize member 'y' under profile 'std::init'}}
};

struct OutOfLine {
  int x; // expected-note {{member 'x' declared here}}
  OutOfLine();
};
OutOfLine::OutOfLine() {} // expected-error {{constructor does not initialize member 'x' under profile 'std::init'}}

template <typename T>
struct Tmpl {
  T x; // expected-note {{member 'x' declared here}}
  Tmpl() {} // expected-error {{constructor does not initialize member 'x' under profile 'std::init'}}
};
template struct Tmpl<int>; // expected-note {{in instantiation of member function 'Tmpl<int>::Tmpl' requested here}}

// A delegating constructor relies on its target, which does initialize the
// member, so nothing fires.
struct Delegating {
  int x;
  Delegating() : Delegating(0) {}
  Delegating(int v) : x(v) {}
};

struct SuppressedCtor {
  int x;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] SuppressedCtor() {}
};

struct SuppressedByRule {
  int x;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ctor_uninit_member")]] SuppressedByRule() {}
};

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(std::init)]] SuppressedClass {
  int x;
  SuppressedClass() {}
};
