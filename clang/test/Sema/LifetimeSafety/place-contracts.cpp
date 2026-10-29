// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wlifetime-safety-experimental-place-contract -Wno-dangling -verify %s

// Methods of a class that declares places (or uses `invalidates`) are checked
// against their own contracts:
//  - an interior invalidation of `*this` must be declared by the method's
//    [[clang::invalidates]];
//  - a value returned under [[clang::lifetimebound(place)]] must borrow only
//    that place.

#include "Inputs/lifetime-analysis.h"

template <typename... Ts> void use(const Ts &...);

namespace Contracts {
struct [[clang::places(keys, values)]] T {
  [[clang::place(keys)]] std::vector<std::string> ks_;
  [[clang::place(values)]] std::vector<int> vs_;
  std::vector<int> other_; // not bound to a place
  int n_;

  void Clear() [[clang::invalidates(interior)]];
  void Intern() [[clang::invalidates(keys)]];
  const char *AnyInterior() const [[clang::lifetimebound(interior)]];

  // ---- Invalidation contracts ----
  void Fine() [[clang::invalidates(keys)]] { ks_.push_back("x"); }
  void FineInterior() [[clang::invalidates(interior)]] {
    ks_.push_back("x");
    vs_.push_back(1);
    other_.push_back(1);
    Clear();
  }
  void FineBoth() [[clang::invalidates(keys, values)]] {
    ks_.push_back("x");
    vs_.push_back(1);
  }
  void FineNoInvalidation() { n_ = 1; use(ks_); }

  void WrongPlace() [[clang::invalidates(keys)]] {
    vs_.push_back(1); // expected-warning {{invalidates place 'values' of 'this' but 'WrongPlace' only declares 'invalidates(keys)'}}
  }
  void NoContract() {
    ks_.push_back("x"); // expected-warning {{invalidates place 'keys' of 'this' but 'NoContract' is not annotated 'invalidates'}}
  }
  void UnboundField() [[clang::invalidates(keys)]] {
    other_.push_back(1); // expected-warning {{invalidates field 'other_' of 'this' but 'UnboundField' only declares 'invalidates(keys)'}}
  }
  void CalleeContract() [[clang::invalidates(keys)]] {
    Intern(); // OK: {keys} is declared.
    Clear();  // expected-warning {{invalidates the interior of 'this' but 'CalleeContract' only declares 'invalidates(keys)'}}
  }
  void CalleeContractNone() {
    Intern(); // expected-warning {{invalidates place 'keys' of 'this' but 'CalleeContractNone' is not annotated 'invalidates'}}
  }
  void ThroughPointer() {
    T *self = this;
    self->ks_.push_back("x"); // expected-warning {{invalidates place 'keys' of 'this' but 'ThroughPointer' is not annotated 'invalidates'}}
  }
  void OtherObjectIsFine(T &o) {
    o.ks_.push_back("x"); // OK: not *this.
  }
  void ClearAll() [[clang::invalidates(interior)]] { ks_.clear(); vs_.clear(); }

  // ---- Borrow contracts ----
  std::string_view Key(int i) [[clang::lifetimebound(keys)]] {
    return ks_[i];
  }
  const int &Value(int i) [[clang::lifetimebound(values)]] {
    return vs_[i];
  }
  const char *Interior() [[clang::lifetimebound(interior)]] {
    return ks_[0].data();
  }
  const T &Self() const [[clang::lifetimebound]] { return *this; }
  const int &Unbound(int i) [[clang::lifetimebound(keys)]] {
    return other_[i]; // OK: a sub-object not bound to any place only dies with the object.
  }

  const int &BadPlace(int i) [[clang::lifetimebound(keys)]] { // expected-warning {{returned value borrows place 'values' of 'this' but 'BadPlace' is annotated 'lifetimebound(keys)'}}
    return vs_[i]; // expected-note {{returned here}}
  }
  const T &BadObject() const [[clang::lifetimebound(interior)]] { // expected-warning {{returned value borrows the object itself of 'this' but 'BadObject' is annotated 'lifetimebound(interior)'}}
    return *this; // expected-note {{returned here}}
  }
  const char *BadWildcard() const [[clang::lifetimebound(keys)]] { // expected-warning {{returned value borrows the interior of 'this' but 'BadWildcard' is annotated 'lifetimebound(keys)'}}
    return AnyInterior(); // expected-note {{returned here}}
  }
};

// Constructors and destructors are exempt.
struct [[clang::places(items)]] Ctor {
  [[clang::place(items)]] std::vector<int> items_;
  Ctor() { items_.push_back(1); }
  ~Ctor() { items_.clear(); }
};

// Classes without places or `invalidates` are not checked.
struct NotOptedIn {
  std::vector<int> v_;
  void Push() { v_.push_back(1); }
};

// `invalidates` alone opts a class in.
struct OptedInByInvalidates {
  std::vector<int> v_;
  void Reset() [[clang::invalidates(interior)]] { v_.clear(); }
  void Push() {
    v_.push_back(1); // expected-warning {{invalidates field 'v_' of 'this' but 'Push' is not annotated 'invalidates'}}
  }
};
} // namespace Contracts
