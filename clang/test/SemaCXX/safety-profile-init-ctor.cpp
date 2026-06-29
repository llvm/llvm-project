// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

struct WithCtor { WithCtor(); };
struct Inner { int y; };
struct InnerMarked { int y [[uninit]]; };
struct InnerMixed { int a [[uninit]]; int b; };

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
  int x [[uninit]];
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

// A member whose type's only indeterminate scalar is [[uninit]] is
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

struct Base { int b; };

struct UninitBase : Base { // expected-note {{base class 'Base' declared here}}
  UninitBase() {} // expected-error {{constructor does not initialize base class 'Base' under profile 'std::init'}}
};

struct BaseInitParen : Base {
  BaseInitParen() : Base() {}
};

struct BaseInitBraces : Base {
  BaseInitBraces() : Base{} {}
};

// A base with a user-provided default constructor is trusted.
struct TrustedBaseCtor : WithCtor {
  TrustedBaseCtor() {}
};

// A base whose only indeterminate scalar is [[uninit]]-marked is trusted.
struct MarkedBaseSub : InnerMarked {
  MarkedBaseSub() {}
};

struct MixedBaseMember : Base { // expected-note {{base class 'Base' declared here}}
  int x; // expected-note {{member 'x' declared here}}
  MixedBaseMember() {}
  // expected-error@-1 {{constructor does not initialize member 'x' under profile 'std::init'}}
  // expected-error@-2 {{constructor does not initialize base class 'Base' under profile 'std::init'}}
};

struct MultipleBases : Base, Inner {
  // expected-note@-1 {{base class 'Base' declared here}}
  // expected-note@-2 {{base class 'Inner' declared here}}
  MultipleBases() {}
  // expected-error@-1 {{constructor does not initialize base class 'Base' under profile 'std::init'}}
  // expected-error@-2 {{constructor does not initialize base class 'Inner' under profile 'std::init'}}
};

struct OutOfLineBase : Base { // expected-note {{base class 'Base' declared here}}
  OutOfLineBase();
};
OutOfLineBase::OutOfLineBase() {} // expected-error {{constructor does not initialize base class 'Base' under profile 'std::init'}}

template <typename T>
struct TmplBase : T { // expected-note {{base class 'Base' declared here}}
  TmplBase() {} // expected-error {{constructor does not initialize base class 'Base' under profile 'std::init'}}
};
template struct TmplBase<Base>; // expected-note {{in instantiation of member function 'TmplBase<Base>::TmplBase' requested here}}

struct SuppressedBaseByRule : Base {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "ctor_uninit_member")]] SuppressedBaseByRule() {}
};

// A virtual base is the most-derived constructor's responsibility, so leaving
// it uninitialized here is deferred: no diagnostic.
struct VirtualBase : virtual Base {
  VirtualBase() {}
};
