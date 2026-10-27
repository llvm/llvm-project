// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wlifetime-safety-experimental-invalidation -Wno-dangling -verify %s

// Tests for derivation and invalidation contracts over places:
//   [[clang::lifetimebound]] / (interior) / (place)  -- what a result borrows
//   [[clang::invalidates]]   / (interior) / (place)  -- what a call invalidates
//   [[clang::places(...)]] on a class, [[clang::place(p)]] on a field.

#include "Inputs/lifetime-analysis.h"

template <typename... Ts> void use(const Ts &...);

//===----------------------------------------------------------------------===//
// Object vs interior borrows and invalidations on a user type.
//===----------------------------------------------------------------------===//
namespace ObjectVsInterior {
struct Arena {
  char *Alloc(unsigned n) const [[clang::lifetimebound(interior)]];
  const Arena &Self() const [[clang::lifetimebound]];
  void Reset() [[clang::invalidates(interior)]];
  void Destroy() [[clang::invalidates]];
};

void InteriorBorrowDiesOnReset(Arena &a) { // expected-warning {{parameter 'a' is later invalidated}}
  char *p = a.Alloc(8);                    // expected-note {{result of call to 'Alloc' aliases the storage of parameter 'a'}}
  const Arena &s = a.Self();
  a.Reset();                               // expected-note {{parameter 'a' is invalidated here}}
  s.Alloc(1);                              // OK: object borrow.
  p[0] = 1;                                // expected-note {{later used here}}
}

void ObjectBorrowDiesOnDestroy(Arena &a) { // expected-warning {{parameter 'a' is later invalidated}}
  const Arena *s = &a.Self();              // expected-note {{result of call to 'Self' aliases the storage of parameter 'a'}}
  a.Destroy();                             // expected-note {{parameter 'a' is invalidated here}}
  s->Alloc(1);                             // expected-note {{later used here}}
}

void LocalArena() {
  Arena a;
  char *p = a.Alloc(8); // expected-warning {{local variable 'a' is later invalidated}} \
                        // expected-note {{result of call to 'Alloc' aliases the storage of local variable 'a'}}
  Arena *pa = &a;
  a.Reset();            // expected-note {{local variable 'a' is invalidated here}}
  pa->Alloc(1);         // OK
  use(*p);              // expected-note {{later used here}}
}

void NotUsedAfterIsOk(Arena &a) {
  char *p = a.Alloc(8);
  use(*p);
  a.Reset();
}
} // namespace ObjectVsInterior

//===----------------------------------------------------------------------===//
// `invalidates` on parameters.
//===----------------------------------------------------------------------===//
namespace OnParameters {
void Grow(std::vector<int> &v [[clang::invalidates(interior)]]);
void PushBack(std::vector<int> *v [[clang::invalidates(interior)]]) {
  v->push_back(42);
  use(v); // OK: the vector object survives its own modification.
}
void Release(std::vector<int> *v [[clang::invalidates]]);

void CallerOfGrow() {
  std::vector<int> v;
  auto it = v.begin(); // expected-warning {{local variable 'v' is later invalidated}} \
                       // expected-note {{result of call to 'begin' aliases the storage of local variable 'v'}}
  std::vector<int> *pv = &v;
  Grow(v);             // expected-note {{local variable 'v' is invalidated here}}
  use(*pv);            // OK
  use(*it);            // expected-note {{later used here}}
}

void CallerOfPushBack() {
  std::vector<int> v;
  auto it = v.begin(); // expected-warning {{local variable 'v' is later invalidated}} \
                       // expected-note {{result of call to 'begin' aliases the storage of local variable 'v'}}
  PushBack(&v);        // expected-note {{local variable 'v' is invalidated here}}
  use(*it);            // expected-note {{later used here}}
}

void CallerOfRelease() {
  std::vector<int> *v = new std::vector<int>; // expected-warning {{allocated object is later invalidated}}
  std::vector<int> *alias = v;                // expected-note {{local variable 'v' aliases the storage of allocated object}}
  Release(v);                                 // expected-note {{allocated object is invalidated here}}
  use(*alias);                                // expected-note {{later used here}}
}

void ReleaseParam(std::vector<int> *v) { // expected-warning {{parameter 'v' is later invalidated}}
  Release(v);                            // expected-note {{parameter 'v' is invalidated here}}
  use(*v);                               // expected-note {{later used here}}
}

void GrowDoesNotInvalidateSiblings() {
  std::vector<int> v, w;
  auto it = w.begin();
  Grow(v);
  use(*it); // OK
}
} // namespace OnParameters

//===----------------------------------------------------------------------===//
// Named places.
//===----------------------------------------------------------------------===//
namespace NamedPlaces {
struct [[clang::places(keys, values)]] SymbolTable {
  [[clang::place(keys)]] std::vector<std::string> key_storage_;
  [[clang::place(values)]] std::vector<int> value_storage_;

  std::string_view Key(int i) const [[clang::lifetimebound(keys)]];
  std::span<const int> Values() const [[clang::lifetimebound(values)]];
  const SymbolTable &Self() const [[clang::lifetimebound]];
  const char *AnyInterior() const [[clang::lifetimebound(interior)]];

  int Intern(std::string_view key) [[clang::invalidates(keys)]];
  void Clear() [[clang::invalidates(interior)]];
  void Shrink() [[clang::invalidates(keys, values)]];
};

void PlaceSensitive(SymbolTable &t) { // expected-warning {{parameter 't' is later invalidated}}
  std::string_view k = t.Key(0);      // expected-note {{result of call to 'Key' aliases the storage of parameter 't'}}
  std::span<const int> v = t.Values();
  const SymbolTable &s = t.Self();
  t.Intern("x");                      // expected-note {{parameter 't' is invalidated here}}
  use(v);                             // OK: a different place.
  s.Values();                         // OK: object borrow.
  use(k);                             // expected-note {{later used here}}
}

void ClearKillsEveryPlace(SymbolTable &t) { // expected-warning {{parameter 't' is later invalidated}}
  std::span<const int> v = t.Values();      // expected-note {{result of call to 'Values' aliases the storage of parameter 't'}}
  t.Clear();                                // expected-note {{parameter 't' is invalidated here}}
  use(v);                                   // expected-note {{later used here}}
}

void SeveralPlaces(SymbolTable &t) { // expected-warning {{parameter 't' is later invalidated}}
  std::span<const int> v = t.Values(); // expected-note {{result of call to 'Values' aliases the storage of parameter 't'}}
  t.Shrink();                          // expected-note {{parameter 't' is invalidated here}}
  use(v);                              // expected-note {{later used here}}
}

// A borrow of an unspecified interior may be of any place.
void WildcardBorrowIsConservative(SymbolTable &t) { // expected-warning {{parameter 't' is later invalidated}}
  const char *p = t.AnyInterior();                  // expected-note {{result of call to 'AnyInterior' aliases the storage of parameter 't'}}
  t.Intern("x");                                    // expected-note {{parameter 't' is invalidated here}}
  use(*p);                                          // expected-note {{later used here}}
}

// Direct access to a bound field belongs to its place.
void FieldBinding(SymbolTable &t) {         // expected-warning {{parameter 't' is later invalidated}}
  auto ik = t.key_storage_.begin();         // expected-note {{result of call to 'begin' aliases the storage of field 'key_storage_' of parameter 't'}}
  auto iv = t.value_storage_.begin();
  std::vector<std::string> *pk = &t.key_storage_;
  t.Intern("x");                            // expected-note {{parameter 't' is invalidated here}}
  use(*iv);                                 // OK: a different place.
  use(*pk);                                 // OK: a sub-object, not interior storage.
  use(*ik);                                 // expected-note {{later used here}}
}

void LocalTable() {
  SymbolTable t;
  std::string_view k = t.Key(0); // expected-warning {{local variable 't' is later invalidated}} \
                                 // expected-note {{result of call to 'Key' aliases the storage of local variable 't'}}
  t.Intern("x");                 // expected-note {{local variable 't' is invalidated here}}
  use(k);                        // expected-note {{later used here}}
}

std::string_view FirstKey(const SymbolTable &t [[clang::lifetimebound(keys)]]);
void PlaceOnParameter(SymbolTable &t) { // expected-warning {{parameter 't' is later invalidated}}
  std::string_view k = FirstKey(t);     // expected-note {{result of call to 'FirstKey' aliases the storage of parameter 't'}}
  t.Intern("x");                        // expected-note {{parameter 't' is invalidated here}}
  use(k);                               // expected-note {{later used here}}
}
void PlaceOnParameterSibling(SymbolTable &t) {
  std::span<const int> v = t.Values();
  t.Intern("x"); // 'keys' only.
  use(v);        // OK
}
} // namespace NamedPlaces

//===----------------------------------------------------------------------===//
// Invalidations reaching `this`.
//===----------------------------------------------------------------------===//
namespace ThisInvalidation {
struct Holder {
  std::vector<std::string> c;
  std::vector<std::string> d;

  void InvalidateField() {   // expected-warning {{field 'c' of 'this' is later invalidated}}
    auto it = c.begin();     // expected-note {{result of call to 'begin' aliases the storage of field 'c' of 'this'}}
    c.push_back("1");        // expected-note {{field 'c' of 'this' is invalidated here}}
    use(*it);                // expected-note {{later used here}}
  }
  void SiblingFieldIsOk() {
    auto it = c.begin();
    d.push_back("1");
    use(*it);
  }
};

struct [[gsl::Owner]] Container {
  int *begin() [[clang::lifetimebound(interior)]];
  void push_back(int) [[clang::invalidates(interior)]];

  void SelfInvalidation() { // expected-warning {{'this' is later invalidated}}
    int *it = begin();
    push_back(1);           // expected-note {{'this' is invalidated here}}
    use(*it);               // expected-note {{later used here}}
  }
};
} // namespace ThisInvalidation
