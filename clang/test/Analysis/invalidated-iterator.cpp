// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.InvalidatedIterator -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false %s -verify
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.InvalidatedIterator -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_warnIfReached();

void bad_copy_assign_operator1_list(std::list<int> &L1,
                                    const std::list<int> &L2) {
  auto i0 = L1.cbegin();
  L1 = L2;
  *i0; // expected-warning{{Invalidated iterator accessed}}
  clang_analyzer_warnIfReached();
}

void bad_copy_assign_operator1_vector(std::vector<int> &V1,
                                      const std::vector<int> &V2) {
  auto i0 = V1.cbegin();
  V1 = V2;
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_copy_assign_operator1_deque(std::deque<int> &D1,
                                     const std::deque<int> &D2) {
  auto i0 = D1.cbegin();
  D1 = D2;
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_copy_assign_operator1_forward_list(std::forward_list<int> &FL1,
                                            const std::forward_list<int> &FL2) {
  auto i0 = FL1.cbegin();
  FL1 = FL2;
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_assign1_list(std::list<int> &L, int n) {
  auto i0 = L.cbegin();
  L.assign(10, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_assign1_vector(std::vector<int> &V, int n) {
  auto i0 = V.cbegin();
  V.assign(10, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_assign1_deque(std::deque<int> &D, int n) {
  auto i0 = D.cbegin();
  D.assign(10, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_assign1_forward_list(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin();
  FL.assign(10, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_clear1_list(std::list<int> &L) {
  auto i0 = L.cend();
  L.clear();
  --i0; // no-warning
}

void bad_clear1_list(std::list<int> &L) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.clear();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_clear1_vector(std::vector<int> &V) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.clear();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_clear1_vector_decr(std::vector<int> &V) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.clear();
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_clear1_deque(std::deque<int> &D) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.clear();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_clear1_deque_decr(std::deque<int> &D) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.clear();
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_push_back1_list(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.push_back(n);
  *i0; // no-warning
  --i1; // no-warning
}

void good_push_back1_vector(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.push_back(n);
  *i0; // no-warning
}

void bad_push_back1_vector(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.push_back(n);
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_push_back1_deque(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.push_back(n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_push_back1_deque_decr(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.push_back(n);
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_emplace_back1_list(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.emplace_back(n);
  *i0; // no-warning
  --i1; // no-warning
}

void good_emplace_back1_vector(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.emplace_back(n);
  *i0; // no-warning
}

void bad_emplace_back1_vector(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend();
  V.emplace_back(n);
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_emplace_back1_deque(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.emplace_back(n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_emplace_back1_deque_decr(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.emplace_back(n);
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_back1_list(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend(), i2 = i1--;
  L.pop_back();
  *i0; // no-warning
  *i2; // no-warning
}

void bad_pop_back1_list(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend(), i2 = i1--;
  L.pop_back();
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_back1_vector(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend(), i2 = i1--;
  V.pop_back();
  *i0; // no-warning
}

void bad_pop_back1_vector(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend(), i2 = i1--;
  V.pop_back();
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_pop_back1_vector_decr(std::vector<int> &V, int n) {
  auto i0 = V.cbegin(), i1 = V.cend(), i2 = i1--;
  V.pop_back();
  --i2; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_back1_deque(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend(), i2 = i1--;
  D.pop_back();
  *i0; // no-warning
}

void bad_pop_back1_deque(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend(), i2 = i1--;
  D.pop_back();
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_pop_back1_deque_decr(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend(), i2 = i1--;
  D.pop_back();
  --i2; // expected-warning{{Invalidated iterator accessed}}
}

void good_push_front1_list(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.push_front(n);
  *i0; // no-warning
  --i1; // no-warning
}

void bad_push_front1_deque(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.push_front(n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_push_front1_deque_decr(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.push_front(n);
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_push_front1_forward_list(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = FL.cend();
  FL.push_front(n);
  *i0; // no-warning
}

void good_emplace_front1_list(std::list<int> &L, int n) {
  auto i0 = L.cbegin(), i1 = L.cend();
  L.emplace_front(n);
  *i0; // no-warning
  --i1; // no-warning
}

void bad_emplace_front1_deque(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.emplace_front(n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_emplace_front1_deque_decr(std::deque<int> &D, int n) {
  auto i0 = D.cbegin(), i1 = D.cend();
  D.emplace_front(n);
  --i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_emplace_front1_forward_list(std::forward_list<int> &FL, int n) {
  auto i0 = FL.cbegin(), i1 = FL.cend();
  FL.emplace_front(n);
  *i0; // no-warning
}

void good_pop_front1_list(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.pop_front();
  *i1; // no-warning
}

void bad_pop_front1_list(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.pop_front();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_front1_deque(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.pop_front();
  *i1; // no-warning
}

void bad_pop_front1_deque(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.pop_front();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_pop_front1_forward_list(std::forward_list<int> &FL, int n) {
  auto i1 = FL.cbegin(), i0 = i1++;
  FL.pop_front();
  *i1; // no-warning
}

void bad_pop_front1_forward_list(std::forward_list<int> &FL, int n) {
  auto i1 = FL.cbegin(), i0 = i1++;
  FL.pop_front();
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert1_list1(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.insert(i1, n);
  *i0; // no-warning
  *i1; // no-warning
}

void good_insert1_list2(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  i1 = L.insert(i1, n);
  *i1; // no-warning
}

void good_insert1_vector1(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, n);
  *i0; // no-warning
}

void good_insert1_vector2(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  i1 = V.insert(i1, n);
  *i1; // no-warning
}

void bad_insert1_vector(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, n);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert1_deque(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  i0 = D.insert(i1, n);
  *i0; // no-warning
}

void bad_insert1_deque1(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.insert(i1, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_insert1_deque2(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.insert(i1, n);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert2_list1(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.insert(i1, std::move(n));
  *i0; // no-warning
  *i1; // no-warning
}

void good_insert2_list2(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  i1 = L.insert(i1, std::move(n));
  *i1; // no-warning
}

void good_insert2_vector1(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, std::move(n));
  *i0; // no-warning
}

void good_insert2_vector2(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  i1 = V.insert(i1, std::move(n));
  *i1; // no-warning
}

void bad_insert2_vector(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, std::move(n));
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert2_deque(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  i1 = D.insert(i1, std::move(n));
  *i1; // no-warning
}

void bad_insert2_deque1(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.insert(i1, std::move(n));
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_insert2_deque2(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.insert(i1, std::move(n));
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert3_list1(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.insert(i1, 10, n);
  *i0; // no-warning
  *i1; // no-warning
}

void good_insert3_list2(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  i1 = L.insert(i1, 10, n);
  *i1; // no-warning
}

void good_insert3_vector1(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, 10, n);
  *i0; // no-warning
}

void good_insert3_vector2(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  i1 = V.insert(i1, 10, n);
  *i1; // no-warning
}

void bad_insert3_vector(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, 10, n);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert3_deque(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  i1 = D.insert(i1, 10, std::move(n));
  *i1; // no-warning
}

void bad_insert3_deque1(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.insert(i1, 10, std::move(n));
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_insert3_deque2(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.insert(i1, 10, std::move(n));
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert4_list1(std::list<int> &L1, std::list<int> &L2, int n) {
  auto i1 = L1.cbegin(), i0 = i1++;
  L1.insert(i1, L2.cbegin(), L2.cend());
  *i0; // no-warning
  *i1; // no-warning
}

void good_insert4_list2(std::list<int> &L1, std::list<int> &L2, int n) {
  auto i1 = L1.cbegin(), i0 = i1++;
  i1 = L1.insert(i1, L2.cbegin(), L2.cend());
  *i1; // no-warning
}

void good_insert4_vector1(std::vector<int> &V1, std::vector<int> &V2, int n) {
  auto i1 = V1.cbegin(), i0 = i1++;
  V1.insert(i1, V2.cbegin(), V2.cend());
  *i0; // no-warning
}

void good_insert4_vector2(std::vector<int> &V1, std::vector<int> &V2, int n) {
  auto i1 = V1.cbegin(), i0 = i1++;
  i1 = V1.insert(i1, V2.cbegin(), V2.cend());
  *i1; // no-warning
}

void bad_insert4_vector(std::vector<int> &V1, std::vector<int> &V2, int n) {
  auto i1 = V1.cbegin(), i0 = i1++;
  V1.insert(i1, V2.cbegin(), V2.cend());
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert4_deque(std::deque<int> &D1, std::deque<int> &D2, int n) {
  auto i1 = D1.cbegin(), i0 = i1++;
  i1 = D1.insert(i1, D2.cbegin(), D2.cend());
  *i1; // no-warning
}

void bad_insert4_deque1(std::deque<int> &D1, std::deque<int> &D2, int n) {
  auto i1 = D1.cbegin(), i0 = i1++;
  D1.insert(i1, D2.cbegin(), D2.cend());
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_insert4_deque2(std::deque<int> &D1, std::deque<int> &D2, int n) {
  auto i1 = D1.cbegin(), i0 = i1++;
  D1.insert(i1, D2.cbegin(), D2.cend());
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert5_list1(std::list<int> &L) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.insert(i1, {1, 2, 3, 4});
  *i0; // no-warning
  *i1; // no-warning
}

void good_insert5_list2(std::list<int> &L) {
  auto i1 = L.cbegin(), i0 = i1++;
  i1 = L.insert(i1, {1, 2, 3, 4});
  *i1; // no-warning
}

void good_insert5_vector1(std::vector<int> &V) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, {1, 2, 3, 4});
  *i0; // no-warning
}

void good_insert5_vector2(std::vector<int> &V) {
  auto i1 = V.cbegin(), i0 = i1++;
  i1 = V.insert(i1, {1, 2, 3, 4});
  *i1; // no-warning
}

void bad_insert5_vector(std::vector<int> &V) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.insert(i1, {1, 2, 3, 4});
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_insert5_deque(std::deque<int> &D) {
  auto i1 = D.cbegin(), i0 = i1++;
  i1 = D.insert(i1, {1, 2, 3, 4});
  *i1; // no-warning
}

void bad_insert5_deque1(std::deque<int> &D) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.insert(i1, {1, 2, 3, 4});
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_insert5_deque2(std::deque<int> &D) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.insert(i1, {1, 2, 3, 4});
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_emplace1_list(std::list<int> &L, int n) {
  auto i1 = L.cbegin(), i0 = i1++;
  L.emplace(i1, n);
  *i0; // no-warning
  *i1; // no-warning
}

void good_emplace1_vector(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.emplace(i1, n);
  *i0; // no-warning
}

void bad_emplace1_vector(std::vector<int> &V, int n) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.emplace(i1, n);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_emplace1_deque1(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.emplace(i1, n);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_emplace1_deque2(std::deque<int> &D, int n) {
  auto i1 = D.cbegin(), i0 = i1++;
  D.emplace(i1, n);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase1_list1(std::list<int> &L) {
  auto i2 = L.cbegin(), i0 = i2++, i1 = i2++;
  L.erase(i1);
  *i0; // no-warning
  *i2; // no-warning
}

void good_erase1_list2(std::list<int> &L) {
  auto i0 = L.cbegin();
  i0 = L.erase(i0);
  *i0; // no-warning
}

void bad_erase1_list(std::list<int> &L) {
  auto i0 = L.cbegin();
  L.erase(i0);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase1_vector1(std::vector<int> &V) {
  auto i2 = V.cbegin(), i0 = i2++, i1 = i2++;
  V.erase(i1);
  *i0; // no-warning
}

void good_erase1_vector2(std::vector<int> &V) {
  auto i0 = V.cbegin();
  i0 = V.erase(i0);
  *i0; // no-warning
}

void bad_erase1_vector1(std::vector<int> &V) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.erase(i0);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase1_vector2(std::vector<int> &V) {
  auto i1 = V.cbegin(), i0 = i1++;
  V.erase(i0);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase1_deque(std::deque<int> &D) {
  auto i0 = D.cbegin();
  i0 = D.erase(i0);
  *i0; // no-warning
}

void bad_erase1_deque1(std::deque<int> &D) {
  auto i2 = D.cbegin(), i0 = i2++, i1 = i2++;
  D.erase(i1);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase1_deque2(std::deque<int> &D) {
  auto i2 = D.cbegin(), i0 = i2++, i1 = i2++;
  D.erase(i1);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase1_deque3(std::deque<int> &D) {
  auto i2 = D.cbegin(), i0 = i2++, i1 = i2++;
  D.erase(i1);
  *i2; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase2_list1(std::list<int> &L) {
  auto i3 = L.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  L.erase(i1, i3);
  *i0; // no-warning
  *i3; // no-warning
}

void good_erase2_list2(std::list<int> &L) {
  auto i2 = L.cbegin(), i0 = i2++, i1 = i2++;
  i0 = L.erase(i0, i2);
  *i0; // no-warning
}

void bad_erase2_list1(std::list<int> &L) {
  auto i2 = L.cbegin(), i0 = i2++, i1 = i2++;
  L.erase(i0, i2);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase2_list2(std::list<int> &L) {
  auto i2 = L.cbegin(), i0 = i2++, i1 = i2++;
  L.erase(i0, i2);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase2_vector1(std::vector<int> &V) {
  auto i3 = V.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;;
  V.erase(i1, i3);
  *i0; // no-warning
}

void good_erase2_vector2(std::vector<int> &V) {
  auto i2 = V.cbegin(), i0 = i2++, i1 = i2++;
  i0 = V.erase(i0, i2);
  *i0; // no-warning
}

void bad_erase2_vector1(std::vector<int> &V) {
  auto i2 = V.cbegin(), i0 = i2++, i1 = i2++;
  V.erase(i0, i2);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase2_vector2(std::vector<int> &V) {
  auto i2 = V.cbegin(), i0 = i2++, i1 = i2++;
  V.erase(i0, i2);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase2_vector3(std::vector<int> &V) {
  auto i2 = V.cbegin(), i0 = i2++, i1 = i2++;
  V.erase(i0, i2);
  *i2; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase2_deque(std::deque<int> &D) {
  auto i2 = D.cbegin(), i0 = i2++, i1 = i2++;
  i0 = D.erase(i0, i2);
  *i0; // no-warning
}

void bad_erase2_deque1(std::deque<int> &D) {
  auto i3 = D.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  D.erase(i1, i3);
  *i0; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase2_deque2(std::deque<int> &D) {
  auto i3 = D.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  D.erase(i1, i3);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase2_deque3(std::deque<int> &D) {
  auto i3 = D.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  D.erase(i1, i3);
  *i2; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase2_deque4(std::deque<int> &D) {
  auto i3 = D.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  D.erase(i1, i3);
  *i3; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase_after1_forward_list1(std::forward_list<int> &FL) {
  auto i2 = FL.cbegin(), i0 = i2++, i1 = i2++;
  FL.erase_after(i0);
  *i0; // no-warning
  *i2; // no-warning
}

void good_erase_after1_forward_lis2(std::forward_list<int> &FL) {
  auto i1 = FL.cbegin(), i0 = i1++;
  i1 = FL.erase_after(i0);
  *i1; // no-warning
}

void bad_erase_after1_forward_list(std::forward_list<int> &FL) {
  auto i1 = FL.cbegin(), i0 = i1++;
  FL.erase_after(i0);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void good_erase_after2_forward_list1(std::forward_list<int> &FL) {
  auto i3 = FL.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  FL.erase_after(i0, i3);
  *i0; // no-warning
  *i3; // no-warning
}

void good_erase_after2_forward_list2(std::forward_list<int> &FL) {
  auto i3 = FL.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  i2 = FL.erase_after(i0, i3);
  *i2; // no-warning
}

void bad_erase_after2_forward_list1(std::forward_list<int> &FL) {
  auto i3 = FL.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  FL.erase_after(i0, i3);
  *i1; // expected-warning{{Invalidated iterator accessed}}
}

void bad_erase_after2_forward_list2(std::forward_list<int> &FL) {
  auto i3 = FL.cbegin(), i0 = i3++, i1 = i3++, i2 = i3++;
  FL.erase_after(i0, i3);
  *i2; // expected-warning{{Invalidated iterator accessed}}
}
