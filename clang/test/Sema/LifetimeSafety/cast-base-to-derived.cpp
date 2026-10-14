// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify %s

struct Base {};
struct Derived : Base {};
Derived *DowncastReturnsStackAddress() {
  Derived d;
  Base *b = &d;
  // expected-warning@-1 {{stack memory associated with local variable 'd' is returned}}
  return static_cast<Derived *>(b); // expected-note {{returned here}}
}

// A downcast that does not outlive the pointee is fine.
void use(Derived *);
void DowncastWithinScopeIsOk() {
  Derived d;
  Base *b = &d;
  use(static_cast<Derived *>(b)); // ok
}

struct RefBase {};
struct RefDerived : RefBase {};
RefDerived *DowncastReferenceReturnsStackAddress() {
  RefDerived d;
  RefBase &b = d;
  // expected-warning@-1 {{stack memory associated with local variable 'd' is returned}}
  RefDerived &r = static_cast<RefDerived &>(b);
  return &r;      // expected-note {{returned here}}
}

void use(Derived &);
void DowncastReferenceWithinScopeIsOk() {
  Derived d;
  Base &b = d;
  use(static_cast<Derived &>(b)); // ok
}

// ****************************************************************************
// Different origin shapes for BaseToDerived and DerivedToBase casts.
// ****************************************************************************

struct [[gsl::Pointer]] GslBase {};
struct [[gsl::Pointer]] GslDerived : GslBase {};

GslDerived *MatchingAnnotationsStillDetected() {
  GslDerived d;
  GslBase *b = &d;
  // expected-warning@-1 {{stack memory associated with local variable 'd' is returned}}
  return static_cast<GslDerived *>(b); // expected-note {{returned here}}
}

// Base is a gsl::Pointer but the derived is not.
// They have different origin shapes, thus the borrow is dropped, thus no dangling is reported.
struct [[gsl::Pointer]] GslOnlyBase {};
struct PlainFromGsl : GslOnlyBase {};

PlainFromGsl *BaseAnnotatedMismatchDropsBorrow() {
  PlainFromGsl d;
  GslOnlyBase *b = &d;                   // FN: See comment above.
  return static_cast<PlainFromGsl *>(b); // no note
}

// Mismatch the other way: derived is a gsl::Pointer, base is not.
// Same outcome.
struct PlainBase2 {};
struct [[gsl::Pointer]] GslOnlyDerived : PlainBase2 {};

GslOnlyDerived *DerivedAnnotatedMismatchDropsBorrow() {
  GslOnlyDerived d;
  PlainBase2 *b = &d;                      // FN: See comment above.
  return static_cast<GslOnlyDerived *>(b); // no note
}
