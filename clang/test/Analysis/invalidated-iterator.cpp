// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.InvalidatedIterator -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false %s -verify
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.InvalidatedIterator -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

#include "Inputs/system-header-simulator-cxx.h"

void bad_copy_assign_operator_list1(std::list<int> &L1,
                                    const std::list<int> &L2) {
  auto i0 = L1.cbegin();
  L1 = L2;
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_copy_assign_operator_vector1(std::vector<int> &V1,
                                      const std::vector<int> &V2) {
  auto i0 = V1.cbegin();
  V1 = V2;
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_copy_assign_operator_deque1(std::deque<int> &D1,
                                     const std::deque<int> &D2) {
  auto i0 = D1.cbegin();
  D1 = D2;
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_copy_assign_operator_forward_list1(std::forward_list<int> &FL1,
                                            const std::forward_list<int> &FL2) {
  auto i0 = FL1.cbegin();
  FL1 = FL2;
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_assign_list1(std::list<int> &L, int n) {
  auto i0 = L.cbegin();
  L.assign(10, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_assign_vector1(std::vector<int> &V, int n) {
  auto i0 = V.cbegin();
  V.assign(10, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_assign_deque1(std::deque<int> &D, int n) {
  auto i0 = D.cbegin();
  D.assign(10, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_assign_forward_list1(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin();
  FL.assign(10, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_clear_list1(std::list<int> &L) {
  auto i0 = L.cend();
  L.clear();
  --i0; // no-warning
}

void bad_clear_list1(std::list<int> &L) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.clear();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_clear_vector1(std::vector<int> &V) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.clear();
  *i0; // expected-warning{{Invalidated iterator accessed}}
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_clear_deque1(std::deque<int> &D) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.clear();
  *i0; // expected-warning{{Invalidated iterator accessed}}
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_push_back_list1(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.push_back(n);
  *i0; // no-warning
  --i1; // no-warning
}

void good_push_back_vector1(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.push_back(n);
  *i0; // no-warning
}

void bad_push_back_vector1(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.push_back(n);
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_push_back_deque1(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.push_back(n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_emplace_back_list1(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.emplace_back(n);
  *i0; // no-warning
  --i1; // no-warning
}

void good_emplace_back_vector1(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.emplace_back(n);
  *i0; // no-warning
}

void bad_emplace_back_vector1(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.emplace_back(n);
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_emplace_back_deque1(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.emplace_back(n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_back_list1(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend(), i2 = i1--;
  L.pop_back();
  *i0; // no-warning
  *i2; // no-warning
}

void bad_pop_back_list1(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend(), i2 = i1--;
  L.pop_back();
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_back_vector1(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend(), i2 = i1--;
  V.pop_back();
  *i0; // no-warning
}

void bad_pop_back_vector1(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend(), i2 = i1--;
  V.pop_back();
  *i1; // expected-warning{{Invalidated iterator accessed}}
  --i2; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_back_deque1(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend(), i2 = i1--;
  D.pop_back();
  *i0; // no-warning
}

void bad_pop_back_deque1(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend(), i2 = i1--;
  D.pop_back();
  *i1; // expected-warning{{Invalidated iterator accessed}}
  --i2; // expected-warning{{Invalidated iterator accessed}}
}

void good_push_front_list1(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.push_front(n);
  *i0; // no-warning
  --i1; // no-warning
}

void bad_push_front_deque1(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.push_front(n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_push_front_forward_list1(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = FL.cend();
  FL.push_front(n);
  *i0; // no-warning
}

void good_emplace_front_list1(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.emplace_front(n);
  *i0; // no-warning
  --i1; // no-warning
}

void bad_emplace_front_deque1(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.emplace_front(n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_emplace_front_forward_list1(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = FL.cend();
  FL.emplace_front(n);
  *i0; // no-warning
}

void good_pop_front_list1(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.pop_front();
  *i1; // no-warning
}

void bad_pop_front_list1(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.pop_front();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_front_deque1(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.pop_front();
  *i1; // no-warning
}

void bad_pop_front_deque1(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.pop_front();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_front_forward_list1(std::forward_list<int> &FL, int n) {
  auto i1 = FL.cbegin(), i0 = i1++;
  FL.pop_front();
  *i1; // no-warning
}

void bad_pop_front_forward_list1(std::forward_list<int> &FL, int n) {
  auto i1 = FL.cbegin(), i0 = i1++;
  FL.pop_front();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert_list1(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.insert(i1, n);
  *i0; // no-warning
  *i1; // no-warning
}

void good_insert_vector1(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, n);
  *i0; // no-warning
}

void bad_insert_vector1(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, n);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_insert_deque1(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.insert(i1, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_emplace_list1(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.emplace(i1, n);
  *i0; // no-warning
  *i1; // no-warning
}

void good_emplace_vector1(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.emplace(i1, n);
  *i0; // no-warning
}

void bad_emplace_vector1(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.emplace(i1, n);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_emplace_deque1(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.emplace(i1, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase_list1(std::list<int> &L) {
  auto i2 = L.cbegin(), i0 = i2++, i1 = i2++;
  L.erase(i1);
  *i0; // no-warning
  *i2; // no-warning
}

void bad_erase_list1(std::list<int> &L) {
  auto i0 = L.cbegin();
  L.erase(i0);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase_vector1(std::vector<int> &V) {
  auto i2 = V.cbegin(), i0 = i2++, i1 = i2++;
  V.erase(i1);
  *i0; // no-warning
}

void bad_erase_vector1(std::vector<int> &V) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.erase(i0);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase_deque1(std::deque<int> &D) {
  auto i2 = D.cbegin(), i0 = i2++, i1 = i2++;
  D.erase(i1);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  *i1; // expected-warning{{Invalidated iterator accessed}}
  *i2; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase_list2(std::list<int> &L) {
  auto i3 = L.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  L.erase(i1, i3);
  *i0; // no-warning
  *i3; // no-warning
}

void bad_erase_list2(std::list<int> &L) {
  auto i2 = L.cbegin(), i0 = i2++, i1 = i2++;
  L.erase(i0, i2);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase_vector2(std::vector<int> &V) {
  auto i3 = V.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;;
  V.erase(i1, i3);
  *i0; // no-warning
}

void bad_erase_vector2(std::vector<int> &V) {
  auto i2 = V.cbegin(), i0 = i2++, i1 = i2++;
  V.erase(i0, i2);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  *i1; // expected-warning{{Invalidated iterator accessed}}
  *i2; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase_deque2(std::deque<int> &D) {
  auto i3 = D.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  D.erase(i1, i3);
  *i0; // expected-warning{{Invalidated iterator accessed}}
  *i1; // expected-warning{{Invalidated iterator accessed}}
  *i2; // expected-warning{{Invalidated iterator accessed}}
  *i3; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase_after_forward_list1(std::forward_list<int> &FL) {
  auto i2 = FL.cbegin(), i0 = i2++, i1 = i2++;
  FL.erase_after(i0);
  *i0; // no-warning
  *i2; // no-warning
}

void bad_erase_after_forward_list1(std::forward_list<int> &FL) {
  auto i1 = FL.cbegin(), i0 = i1++;
  FL.erase_after(i0);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase_after_forward_list2(std::forward_list<int> &FL) {
  auto i3 = FL.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  FL.erase_after(i0, i3);
  *i0; // no-warning
  *i3; // no-warning
}

void bad_erase_after_forward_list2(std::forward_list<int> &FL) {
  auto i3 = FL.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  FL.erase_after(i0, i3);
  *i1; // expected-warning{{Invalidated iterator accessed}}
  *i2; // expected-warning{{Invalidated iterator accessed}}
}
