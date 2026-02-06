// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify %s

#include "Inputs/lifetime-analysis.h"

bool Bool();

namespace SimpleResize {
void IteratorInvalidAfterResize(int new_size) {
  std::vector<int> v;
  auto it = v.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
  v.resize(new_size);   // expected-note {{invalidated here}}
  *it;                  // expected-note {{later used here}}
}

void IteratorInvalidAfterResizeNotDetected(int new_size) {
  std::vector<int> v;
  // FIXME: We do not currently handle non-member lifetimebound functions to return an interior path.
  // Currently 'it' will have loan with path 'v'. We want it to instead have 'v.*' inorder to be later invalidated.
  auto it = std::begin(v);
  v.resize(new_size);
  *it;
}

void IteratorValidAfterResize(int new_size) {
  std::vector<int> v;
  auto it = v.begin();
  v.resize(new_size);
  it = v.begin();
  if (it != std::end(v)) {
    *it;  // ok
  }
}
}  // namespace SimpleResize

namespace CheckModel {
void IteratorValidAfterCheck() {
  std::vector<int> v;
  auto it = v.begin();
  *it;  // ok
}
}  // namespace CheckModel

namespace PointerToContainer {
std::vector<int>* GetContainerPointer();
void PointerToContainerTest() {
  // FIXME: Use opaque loans.
  std::vector<int>* v = GetContainerPointer();
  auto it = v->begin();
  *it = 0;  // not-ok
}

void PointerToContainerTest(std::vector<int>* v) { // expected-warning {{parameter is later invalidated}}
  auto it = v->begin();
  v->push_back(42); // expected-note {{invalidated here}}
  *it = 0;          // expected-note {{later used here}}
}
}  // namespace PointerToContainer

namespace InvalidateBeforeSwap {
void InvalidateBeforeSwapIterators(std::vector<int> v1, std::vector<int> v2) {
  auto it1 = v1.begin(); // expected-warning {{object whose reference is captured is later invalidated}}
  auto it2 = v2.begin();
  if (it1 == std::end(v1) || it2 == std::end(v2)) return;
  *it1 = 0;     // ok
  *it2 = 0;     // ok
  v1.clear();   // expected-note {{invalidated here}}
  *it1 = 0;     // expected-note {{later used here}}
  // FIXME: Handle invalidating functions like std::swap.
  std::swap(it1, it2);
  *it1 = 0;  // ok
  *it2 = 0;  // not-ok
}

void InvalidateBeforeSwapContainers(std::vector<int> v1, std::vector<int> v2) {
  auto it1 = v1.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
  auto it2 = v2.begin();
  if (it1 == std::end(v1) || it2 == std::end(v2)) return;
  *it1 = 0;     // ok
  *it2 = 0;     // ok
  v1.clear();   // expected-note {{invalidated here}}
  *it1 = 0;     // expected-note {{later used here}}
}
}  // namespace InvalidateBeforeSwap

namespace MergeConditionBasic {
bool A();
bool B();
void SameConditionInvalidatesThenValidatesIterator() {
  std::vector<int> container;
  auto it = container.begin(); // expected-warning {{object whose reference is captured is later invalidated}}
  if (it == container.end()) return;
  const bool a = A();
  if (a) {
    container.clear();  // expected-note {{invalidated here}}
  }
  if (a) {
    it = container.begin();
    if (it == std::end(container)) return;
  }
  *it = 10;  // expected-note {{later used here}}
}
}  // namespace MergeConditionBasic

namespace IteratorWithMultipleContainers {
void MergeWithDifferentContainerValuesIteratorNotInvalidated() {
  std::vector<int> v1, v2, v3;
  auto it = std::find(v1.begin(), v1.end(), 10);
  if (Bool()) {
    it = std::find(v2.begin(), v2.end(), 10);
  } else {
    it = std::find(v3.begin(), v3.end(), 10);
  }
  v1.clear();
  *it = 20;
}

void MergeWithDifferentContainerValuesInvalidated() {
  std::vector<int> v1, v2, v3;
  auto it = std::find(v1.begin(), v1.end(), 10);
  if (Bool()) {
    it = std::find(v2.begin(), v2.end(), 10);  // expected-warning {{object whose reference is captured is later invalidated}}
  } else {
    it = std::find(v3.begin(), v3.end(), 10);
  }
  v2.clear();   // expected-note {{invalidated here}}
  *it = 20;     // expected-note {{later used here}}
}
}  // namespace IteratorWithMultipleContainers

namespace InvalidationInLoops {
void IteratorInvalidationInAForLoop(std::vector<int> v) {
  for (auto it = v.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
       it != std::end(v);
       ++it) {  // expected-note {{later used here}}
    if (Bool()) {
      v.erase(it);  // expected-note {{invalidated here}}
    }
  }
}

void IteratorInvalidationInAWhileLoop(std::vector<int> v) {
  auto it = v.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
  while (it != std::end(v)) {
    if (Bool()) {
      v.erase(it);  // expected-note {{invalidated here}}
    }
    ++it; // expected-note {{later used here}}
  }
}

void IteratorInvalidationInAForeachLoop(std::vector<int> v) {
  for (int& x : v) { // expected-warning {{object whose reference is captured is later invalidated}} \
                     // expected-note {{later used here}}
    if (x % 2 == 0) {
      v.erase(std::find(v.begin(), v.end(), 1)); // expected-note {{invalidated here}}
    }
  }
}
}  // namespace InvalidationInLoops

namespace StdVectorPopBack {
void StdVectorPopBackInvalid(std::vector<int> v) {
  auto it = v.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
  if (it == v.end()) return;
  *it;  // ok
  v.pop_back(); // expected-note {{invalidated here}}
  *it;          // expected-note {{later used here}}
}
}  // namespace StdVectorPopBack


namespace SimpleStdFind {
void IteratorCheckedAfterFind(std::vector<int> v) {
  auto it = std::find(v.begin(), std::end(v), 3);
  if (it != std::end(v)) {
    *it;  // ok
  }
}

void IteratorCheckedAfterFindThenErased(std::vector<int> v) {
  auto it = std::find(v.begin(), std::end(v), 3); // expected-warning {{object whose reference is captured is later invalidated}}
  if (it != std::end(v)) {
    v.erase(it); // expected-note {{invalidated here}}
  }
  *it;  // expected-note {{later used here}}
}
}  // namespace SimpleStdFind

namespace SimpleInsert {
void UseReturnedIteratorAfterInsert(std::vector<int> v) {
  auto it = v.begin();
  it = v.insert(it, 10);
  if (it != std::end(v)) {
    *it;  // ok
  }
}

void UseInvalidIteratorAfterInsert(std::vector<int> v) {
  auto it = v.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
  v.insert(it, 10);         // expected-note {{invalidated here}}
  if (it != std::end(v)) {  // expected-note {{later used here}}
    *it;
  }
}
}  // namespace SimpleInsert

namespace SimpleStdInsert {
void IteratorValidAfterInsert(std::vector<int> v) {
  auto it = v.begin();
  v.insert(it, 0);
  it = v.begin();
  if (it != std::end(v)) {
    *it;  // ok
  }
}

void IteratorInvalidAfterInsert(std::vector<int> v, int value) {
  auto it = v.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
  v.insert(it, 0);      // expected-note {{invalidated here}}
  *it;                  // expected-note {{later used here}}
}
}  // namespace SimpleStdInsert

namespace SimpleInvalidIterators {
void IteratorUsedAfterErase(std::vector<int> v) {
  auto it = v.begin();              // expected-warning {{object whose reference is captured is later invalidated}}
  for (; it != std::end(v); ++it) { // expected-note {{later used here}}
    if (*it > 3) {
      v.erase(it);                  // expected-note {{invalidated here}}
    }
  }
}

void IteratorUsedAfterPushBackParam(std::vector<int>& v) { // expected-warning {{parameter is later invalidated}}
  auto it = v.begin();
  if (it != std::end(v) && *it == 3) {
    v.push_back(4);   // expected-note {{invalidated here}}
  }
  ++it;               // expected-note {{later used here}}
}

void IteratorUsedAfterPushBack(std::vector<int> v) {
  auto it = v.begin(); // expected-warning {{object whose reference is captured is later invalidated}}
  if (it != std::end(v) && *it == 3) {
    v.push_back(4); // expected-note {{invalidated here}}
  }
  ++it;             // expected-note {{later used here}}
}
}  // namespace SimpleInvalidIterators

namespace ElementReferences {
// Testing raw pointers and references to elements, not just iterators.

void ReferenceToVectorElement() {
  std::vector<int> v = {1, 2, 3};
  int& ref = v[0];
  v.push_back(4);
  // FIXME: Detect this as a use of 'ref'.
  // https://github.com/llvm/llvm-project/issues/180187
  ref = 10;
  (void)ref;
}

void PointerToVectorElement() {
  std::vector<int> v = {1, 2, 3};
  int* ptr = &v[0];  // expected-warning {{object whose reference is captured is later invalidated}}
  v.resize(100);     // expected-note {{invalidated here}}
  *ptr = 10;         // expected-note {{later used here}}
}

struct SField { int a; int b;};
void PointerToVectorElementField() {
  std::vector<SField> v = {{1, 2}, {3, 4}};
  int* ptr = &v[0].a;  // expected-warning {{object whose reference is captured is later invalidated}}
  v.resize(100);     // expected-note {{invalidated here}}
  *ptr = 10;         // expected-note {{later used here}}
}

void SelfInvalidatingMap() {
  std::unordered_map<int, int> mp;
  mp[1] = 1;
  mp[2] = mp[1];  // FIXME: Detect this. We are mising a UseFact for the assignment params.
}
} // namespace ElementReferences

namespace Strings {

void append(std::string str) {
  std::string_view view = str;  // expected-warning {{object whose reference is captured is later invalidated}}
  str += "456";                 // expected-note {{invalidated here}}
  (void)view;                   // expected-note {{later used here}}
}
void reassign(std::string str, std::string str2) {
  std::string_view view = str;  // expected-warning {{object whose reference is captured is later invalidated}}
  str = str2;                   // expected-note {{invalidated here}}
  (void)view;                   // expected-note {{later used here}}
}

void append_call(std::string str) {
  std::string_view view = str;  // expected-warning {{object whose reference is captured is later invalidated}}
  str.append("456");            // expected-note {{invalidated here}}
  (void)view;                   // expected-note {{later used here}}
}

void replace_call(std::string str) {
  std::string_view view = str;  // expected-warning {{object whose reference is captured is later invalidated}}
  str.replace(0, 1, "456");     // expected-note {{invalidated here}}
  (void)view;                   // expected-note {{later used here}}
}
} // namespace Strings

// FIXME: This should be diagnosed as use-after-invalidation but with potential move.
void ReassigningAfterMove(std::string str, std::string str2) {
  std::string_view view = str;  // expected-warning {{object whose reference is captured is later invalidated}}
  std::vector<std::string> someStorage;
  someStorage.push_back(std::move(str));
  str = str2;   // expected-note {{invalidated here}}
  (void)view;   // expected-note {{later used here}}
}

namespace ContainersAsFields {
struct S {
  std::vector<std::string> strings1;
  std::vector<std::string> strings2;
};

void Invalidate1Use1IsInvalid() {
  S s;
  auto it = s.strings1.begin(); // expected-warning {{object whose reference is captured is later invalidated}}
  s.strings1.push_back("1");    // expected-note {{invalidated here}}
  *it;                          // expected-note {{later used here}}
}

void Invalidate1Use2IsOk() {
    S s;
    auto it = s.strings1.begin();
    s.strings2.push_back("1");
    *it;
}

void Invalidate1Use2ViaRefIsOk() {
    S s;
    auto it1 = s.strings1.begin();
    auto it2 = s.strings2.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
    auto& strings2 = s.strings2;
    strings2.push_back("1");        // expected-note {{invalidated here}}
    *it1;
    *it2;                           // expected-note {{later used here}}
}

void InvalidateBothInASingleExpression(bool cond) {
    S s;
    auto it1 = s.strings1.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
    auto it2 = s.strings2.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
    auto& both = cond ? s.strings1 : s.strings2;
    both.push_back("1");            // expected-note 2 {{invalidated here}}
    *it1;                           // expected-note {{later used here}}
    *it2;                           // expected-note {{later used here}}
}

void Invalidate1UseSIsOk() {
  S s;
  S* p = &s;
  s.strings2.push_back("1");
  (void)*p;
}

// FIXME: Detect invalidation of fields.
// https://github.com/llvm/llvm-project/issues/180992
struct InvalidateMemberFields {
  InvalidateMemberFields();
  void invalidateField() {
    auto it = container.begin();
    container.push_back("1");
    *it;
  }
  void invalidateFieldRef() {
    auto it = contiainerRef.begin();
    contiainerRef.push_back("1");
    *it;
  }
private:
  std::vector<std::string> container;
  std::vector<std::string>& contiainerRef;
};
} // namespace ContainersAsFields

void PointerToContainerIsOk() {
  std::vector<std::string> s;
  std::vector<std::string>* p = &s;
  p->push_back("1");
  (void)*p;
}

void IteratorFromPointerToContainerIsInvalidated() {
  std::vector<std::string> s;
  std::vector<std::string>* p = &s; // expected-warning {{object whose reference is captured is later invalidated}}
  auto it = p->begin();
  p->push_back("1");                // expected-note {{invalidated here}}
  *it;                              // expected-note {{later used here}}
}

void ChangingRegionOwnedByContainerIsOk() {
  std::vector<std::string> subdirs;
  for (std::string& path : subdirs)
    path = std::string();
}

namespace MapInvalidations {
void MapOperatorBracket() {
    std::unordered_map<int, int> m;
    auto it = m.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
    m[1];                 // expected-note {{invalidated here}}
    *it;                  // expected-note {{later used here}}
}
} // namespace MapInvalidations

namespace NestedContainers {
// FIXME: Maybe come up with a access path representation to detect this.
void InnerIteratorInvalidated() {
  std::vector<std::vector<int>> v;
  v.resize(1);
  // We cannot differentiate between v.* and v.*.*.
  // An annotation system along with lifetimebound could be helpful to describe the paths returned.
  auto it = v[0].begin();
  v[0].push_back(1);
  *it;
}
void InnerIteratorInvalidatedOneUseOtherIsStillBad() {
  std::vector<std::vector<int>> v;
  v.resize(100);
  // We have no way to differentiate between v[0] and v[1] regions.
  auto it = v[0].begin();
  v[1].push_back(1);
  *it;
}

void OuterIteratorNotInvalidated() {
  std::vector<std::vector<int>> v;
  v.resize(1);
  auto it = v.begin();
  v[0].push_back(1);
  it->clear(); // OK
}
} // namespace NestedContainers
