// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety -Wno-dangling -verify %s

#include "Inputs/lifetime-analysis.h"

bool Bool();

namespace SimpleResize {
void IteratorInvalidAfterResize(int new_size) {
  std::vector<int> v;
  auto it = std::begin(v);  // expected-warning {{object whose reference is captured is later invalidated}}
  v.resize(new_size);       // expected-note {{invalidated here}}
  *it;                      // expected-note {{later used here}}
}

void IteratorValidAfterResize(int new_size) {
  std::vector<int> v;
  auto it = std::begin(v);
  v.resize(new_size);
  it = std::begin(v);
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
void PointerToContainerTest(std::vector<int>* v) {
  // FIXME: Handle placeholder loans.
  auto it = v->begin();
  *it = 0;  // not-ok
}
}  // namespace PointerToContainer

namespace InvalidateBeforeSwap {
void InvalidateBeforeSwapIterators(std::vector<int> v1, std::vector<int> v2) {
  auto it1 = std::begin(v1); // expected-warning {{object whose reference is captured is later invalidated}}
  auto it2 = std::begin(v2);
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
  auto it1 = std::begin(v1);  // expected-warning {{object whose reference is captured is later invalidated}}
  auto it2 = std::begin(v2);
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
  for (auto it = std::begin(v);  // expected-warning {{object whose reference is captured is later invalidated}}
       it != std::end(v);
       ++it) {  // expected-note {{later used here}}
    if (Bool()) {
      v.erase(it);  // expected-note {{invalidated here}}
    }
  }
}

void IteratorInvalidationInAWhileLoop(std::vector<int> v) {
  auto it = std::begin(v);  // expected-warning {{object whose reference is captured is later invalidated}}
  while (it != std::end(v)) {
    if (Bool()) {
      v.erase(it);  // expected-note {{invalidated here}}
    }
    ++it; // expected-note {{later used here}}
  }
}

void NoIteratorInvalidationInAWhileLoopErase(std::unordered_map<int, int> mp) {
  auto it = mp.begin();
  while (it != std::end(mp)) {
    if (Bool()) {
      auto next = it;
      ++next;
      mp.erase(it); // Ok. 'next' remains valid.
      it = next;
    }
    else {
      ++it;
    }
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
void StdVectorPopBackDoesNotInvalidateOthers(std::vector<int> v) {
  auto it = v.begin();
  if (it == v.end()) return;
  *it;
  v.pop_back();
  *it;
}
}  // namespace StdVectorPopBack


namespace SimpleStdFind {
void IteratorCheckedAfterFind(std::vector<int> v) {
  auto it = std::find(std::begin(v), std::end(v), 3);
  if (it != std::end(v)) {
    *it;  // ok
  }
}

void IteratorCheckedAfterFindThenErased(std::vector<int> v) {
  auto it = std::find(std::begin(v), std::end(v), 3); // expected-warning {{object whose reference is captured is later invalidated}}
  if (it != std::end(v)) {
    v.erase(it); // expected-note {{invalidated here}}
  }
  *it;  // expected-note {{later used here}}
}
}  // namespace SimpleStdFind

namespace SimpleInsert {
void UseReturnedIteratorAfterInsert(std::vector<int> v) {
  auto it = std::begin(v);
  it = v.insert(it, 10);
  if (it != std::end(v)) {
    *it;  // ok
  }
}

void UseInvalidIteratorAfterInsert(std::vector<int> v) {
  auto it = std::begin(v);  // expected-warning {{object whose reference is captured is later invalidated}}
  v.insert(it, 10);         // expected-note {{invalidated here}}
  if (it != std::end(v)) {  // expected-note {{later used here}}
    *it;
  }
}
}  // namespace SimpleInsert

namespace SimpleStdInsert {
void IteratorValidAfterInsert(std::vector<int> v) {
  auto it = std::begin(v);
  v.insert(it, 0);
  it = std::begin(v);
  if (it != std::end(v)) {
    *it;  // ok
  }
}

void IteratorInvalidAfterInsert(std::vector<int> v, int value) {
  auto it = std::begin(v);  // expected-warning {{object whose reference is captured is later invalidated}}
  v.insert(it, 0);          // expected-note {{invalidated here}}
  *it;                      // expected-note {{later used here}}
}
}  // namespace SimpleStdInsert

namespace SimpleInvalidIterators {
void IteratorUsedAfterErase(std::vector<int> v) {
  auto it = std::begin(v);          // expected-warning {{object whose reference is captured is later invalidated}}
  for (; it != std::end(v); ++it) { // expected-note {{later used here}}
    if (*it > 3) {
      v.erase(it);                  // expected-note {{invalidated here}}
    }
  }
}

// FIXME: Detect this. We currently skip invalidation through ref/pointers to containers.
void IteratorUsedAfterPushBackParam(std::vector<int>& v) {
  auto it = std::begin(v);
  if (it != std::end(v) && *it == 3) {
    v.push_back(4);
  }
  ++it;
}

void IteratorUsedAfterPushBack(std::vector<int> v) {
  auto it = std::begin(v); // expected-warning {{object whose reference is captured is later invalidated}}
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
  int& ref = v[0]; // expected-warning {{object whose reference is captured is later invalidated}}
  v.push_back(4);  // expected-note {{invalidated here}}
  ref = 10;        // expected-note {{later used here}}
  (void)ref;
}

void PointerRefToVectorElement() {
  std::vector<int*> v = {nullptr, nullptr};
  int*& ref = v[0];     // expected-warning {{object whose reference is captured is later invalidated}}
  v.push_back(nullptr); // expected-note {{invalidated here}}
  ref = nullptr;        // expected-note {{later used here}}
}

void PointerToVectorElement() {
  std::vector<int> v = {1, 2, 3};
  int* ptr = &v[0];  // expected-warning {{object whose reference is captured is later invalidated}}
  v.resize(100);     // expected-note {{invalidated here}}
  *ptr = 10;         // expected-note {{later used here}}
}

void SelfInvalidatingMap() {
  std::flat_map<int, std::string> mp;
  // TODO: We do not have a way to differentiate between pointer stability and iterator stability!
  //
  // std::unordered_map and other node-based containers provide pointer/reference stability.
  // Therefore the following is safe in practice.
  // On the other hand, std::flat_map (since C++23) does not provide pointer stability on
  // insertion and following is unsafe for this container.
  mp[1] = "42";
  mp[2]     // expected-note {{invalidated here}}
    = 
    mp[1];  // expected-warning {{object whose reference is captured is later invalidated}} expected-note {{later used here}}
}

void InvalidateErase() {
  std::flat_map<int, std::string> mp;
  // None of these containers provide iterator stability. So following is unsafe:
  auto it = mp.find(3); // expected-warning {{object whose reference is captured is later invalidated}}
  mp.erase(mp.find(4)); // expected-note {{invalidated here}} 
  if (it != mp.end())   // expected-note {{later used here}}
    *it;
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
// FIXME: Make Paths more precise to reason at field granularity.
//        Currently we only detect invalidations to direct declarations and not members.
void Invalidate1Use1IsInvalid() {
  // FIXME: Detect this.
  S s;
  auto it = s.strings1.begin();
  s.strings1.push_back("1");
  *it;
}
void Invalidate1Use2IsOk() {
    S s;
    auto it = s.strings1.begin();
    s.strings2.push_back("1");
    *it;
}void Invalidate1Use2ViaRefIsOk() {
    S s;
    auto it = s.strings2.begin();
    auto& strings2 = s.strings2;
    strings2.push_back("1");
    *it;
}
void Invalidate1UseSIsOk() {
  S s;
  S* p = &s;
  s.strings2.push_back("1");
  (void)*p;
}
void PointerToContainerIsOk() {
  std::vector<std::string> s;
  std::vector<std::string>* p = &s;
  p->push_back("1");
  (void)*p;
}
void IteratorFromPointerToContainerIsInvalidated() {
  // FIXME: Detect this.
  std::vector<std::string> s;
  std::vector<std::string>* p = &s;
  auto it = p->begin();
  p->push_back("1");
  *it;
}
void ChangingRegionOwnedByContainerIsOk() {
  std::vector<std::string> subdirs;
  for (std::string& path : subdirs)
    path = std::string();
}

} // namespace ContainersAsFields

namespace AssociativeContainers {
void SetInsertDoesNotInvalidate() {
  std::set<int> s;
  s.insert(0);
  auto it = s.begin();
  s.insert(2);
  *it;
}

void MapInsertDoesNotInvalidate() {
  std::map<int, int> m;
  auto it = m.begin();
  m.insert({1, 2});
  *it;
}

void MapEmplaceDoesNotInvalidate() {
  std::map<int, int> m;
  auto it = m.begin();
  m.emplace(1, 2);
  *it;
}

void MultisetInsertDoesNotInvalidate() {
  std::multiset<int> s;
  auto it = s.begin();
  s.insert(1);
  *it;
}

void MultimapInsertDoesNotInvalidate() {
  std::multimap<int, int> m;
  auto it = m.begin();
  m.insert({1, 2});
  *it;
}

void SetEraseDoesNotInvalidateOthers() {
  std::set<int> s;
  s.insert(1);
  s.insert(2);
  auto it1 = s.begin();
  auto it2 = it1;
  ++it2;
  s.erase(it2);
  *it1;
}

void SetExtractDoesNotInvalidateOthers() {
  std::set<int> s;
  s.insert(1);
  s.insert(2);
  auto it1 = s.begin();
  auto it2 = it1;
  ++it2;
  s.extract(it2);
  *it1;
}

void SetClearInvalidates() {
  std::set<int> s;
  auto it = s.begin(); // expected-warning {{object whose reference is captured is later invalidated}}
  s.clear(); // expected-note {{invalidated here}}
  *it; // expected-note {{later used here}}
}

void MapClearInvalidates() {
  std::map<int, int> m;
  auto it = m.begin();  // expected-warning {{object whose reference is captured is later invalidated}}
  m.clear(); // expected-note {{invalidated here}}
  *it; // expected-note {{later used here}}
}

void MapSubscriptDoesNotInvalidate() {
  std::map<int, int> m;
  auto it = m.begin();
  m[1];
  *it;
}

void PrintMax(const int& a, const int& b);

void MapSubscriptMultipleCallsDoesNotInvalidate(std::map<int, int> mp, int a, int b) {
    PrintMax(mp[a], mp[b]);
}

void FlatMapSubscriptMultipleCallsInvalidate(std::flat_map<int, int> mp, int a, int b) {
    PrintMax(mp[a], mp[b]); // expected-warning {{object whose reference is captured is later invalidated}} \
                                 // expected-note {{invalidated here}} \
                                 // expected-note {{later used here}}
}

} // namespace AssociativeContainers

namespace lambda_capture_invalidation {
void captured_view_invalidated_by_owner() {
  std::string s = "42";
  std::string_view p = s; // expected-warning {{object whose reference is captured is later invalidated}}
  auto lambda = [=]() { return p; };
  s.push_back('c');  // expected-note {{invalidated here}}
  lambda();  // expected-note {{later used here}}
}

void multiple_captures_one_invalidated() {
  std::string s1 = "a", s2 = "b";
  std::string_view p1 = s1, p2 = s2; // expected-warning {{object whose reference is captured is later invalidated}}
  auto lambda = [=]() { return p1.size() + p2.size(); };
  s1.clear();  // expected-note {{invalidated here}}
  lambda();  // expected-note {{later used here}}
}

// FIXME: By-ref captures flow only the outermost origin, so
// invalidation of the captured view's pointee is not propagated.
void ref_capture_owner_invalidated() {
  std::string s = "42";
  std::string_view p = s;
  auto lambda = [&]() { return p; };
  s.push_back('c');  // invalidates p
  lambda();  // should warn: use-after-invalidate
}

// FIXME: Once inner origins are tracked, this case must remain a no-warning.
// Reassigning `p` through the by-ref capture should invalidate the link to `s`.
void ref_capture_reassigned_to_safe() {
  std::string s = "42", safe = "not modified";
  std::string_view p = s;
  auto lambda = [&]() { return p; };
  p = safe;  // p now points to 'safe', not 's'
  s.push_back('c');  // does not invalidate p anymore
  lambda();  // should not warn
}
} // namespace lambda_capture_invalidation
