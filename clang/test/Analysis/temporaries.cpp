// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-ipa=inlining -verify -w %s

struct Trivial {
  Trivial(int x) : value(x) {}
  int value;
};

struct NonTrivial : public Trivial {
  NonTrivial(int x) : Trivial(x) {}
  ~NonTrivial();
};


Trivial getTrivial() {
  return Trivial(42); // no-warning
}

const Trivial &getTrivialRef() {
  return Trivial(42); // expected-warning {{Address of stack memory associated with temporary object of type 'struct Trivial' returned to caller}}
}


NonTrivial getNonTrivial() {
  return NonTrivial(42); // no-warning
}

const NonTrivial &getNonTrivialRef() {
  return NonTrivial(42); // expected-warning {{Address of stack memory associated with temporary object of type 'struct NonTrivial' returned to caller}}
}

