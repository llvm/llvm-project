// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.MismatchedIterator -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false %s -verify
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.MismatchedIterator -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

#include "Inputs/system-header-simulator-cxx.h"

void good_insert1(std::vector<int> &v, int n) {
  v.insert(v.cbegin(), n); // no-warning
}


void good_insert2(std::vector<int> &v, int len, int n) {
  v.insert(v.cbegin(), len, n); // no-warning
}

void good_insert3(std::vector<int> &v1, std::vector<int> &v2) {
  v1.insert(v1.cbegin(), v2.cbegin(), v2.cend()); // no-warning
}

void good_insert4(std::vector<int> &v, int len, int n) {
  v.insert(v.cbegin(), {n-1, n, n+1}); // no-warning
}

void good_insert_find(std::vector<int> &v, int n, int m) {
  auto i = std::find(v.cbegin(), v.cend(), n);
  v.insert(i, m); // no-warning
}

void good_erase1(std::vector<int> &v) {
  v.erase(v.cbegin()); // no-warning
}

void good_erase2(std::vector<int> &v) {
  v.erase(v.cbegin(), v.cend()); // no-warning
}

void good_emplace(std::vector<int> &v, int n) {
  v.emplace(v.cbegin(), n); // no-warning
}

void good_ctor(std::vector<int> &v) {
  std::vector<int> new_v(v.cbegin(), v.cend()); // no-warning
}

void good_find(std::vector<int> &v, int n) {
  std::find(v.cbegin(), v.cend(), n); // no-warning
}

void good_find_first_of(std::vector<int> &v1, std::vector<int> &v2) {
  std::find_first_of(v1.cbegin(), v1.cend(), v2.cbegin(), v2.cend()); // no-warning
}

void good_copy(std::vector<int> &v1, std::vector<int> &v2, int n) {
  std::copy(v1.cbegin(), v1.cend(), v2.begin()); // no-warning
}

void good_move_find1(std::vector<int> &v1, std::vector<int> &v2, int n) {
  auto i0 = v2.cbegin();
  v1 = std::move(v2);
  std::find(i0, v1.cend(), n); // no-warning
}

void bad_insert1(std::vector<int> &v1, std::vector<int> &v2, int n) {
  v2.insert(v1.cbegin(), n); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_insert2(std::vector<int> &v1, std::vector<int> &v2, int len, int n) {
  v2.insert(v1.cbegin(), len, n); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_insert3(std::vector<int> &v1, std::vector<int> &v2) {
  v2.insert(v1.cbegin(), v2.cbegin(), v2.cend()); // expected-warning{{Container accessed using foreign iterator argument}}
  v1.insert(v1.cbegin(), v1.cbegin(), v2.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
  v1.insert(v1.cbegin(), v2.cbegin(), v1.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
}

void bad_insert4(std::vector<int> &v1, std::vector<int> &v2, int len, int n) {
  v2.insert(v1.cbegin(), {n-1, n, n+1}); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_erase1(std::vector<int> &v1, std::vector<int> &v2) {
  v2.erase(v1.cbegin()); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_erase2(std::vector<int> &v1, std::vector<int> &v2) {
  v2.erase(v2.cbegin(), v1.cend()); // expected-warning{{Container accessed using foreign iterator argument}}
  v2.erase(v1.cbegin(), v2.cend()); // expected-warning{{Container accessed using foreign iterator argument}}
  v2.erase(v1.cbegin(), v1.cend()); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_emplace(std::vector<int> &v1, std::vector<int> &v2, int n) {
  v2.emplace(v1.cbegin(), n); // expected-warning{{Container accessed using foreign iterator argument}}
}

void good_move_find2(std::vector<int> &v1, std::vector<int> &v2, int n) {
  auto i0 = --v2.cend();
  v1 = std::move(v2);
  std::find(i0, v1.cend(), n); // no-warning
}

void good_move_find3(std::vector<int> &v1, std::vector<int> &v2, int n) {
  auto i0 = v2.cend();
  v1 = std::move(v2);
  v2.push_back(n); // expected-warning{{Method called on moved-from object of type 'std::vector'}}
  std::find(v2.cbegin(), i0, n); // no-warning
}

void good_comparison(std::vector<int> &v) {
  if (v.cbegin() == v.cend()) {} // no-warning
}

void bad_ctor(std::vector<int> &v1, std::vector<int> &v2) {
  std::vector<int> new_v(v1.cbegin(), v2.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
}

void bad_find(std::vector<int> &v1, std::vector<int> &v2, int n) {
  std::find(v1.cbegin(), v2.cend(), n); // expected-warning{{Iterators of different containers used where the same container is expected}}
}

void bad_find_first_of(std::vector<int> &v1, std::vector<int> &v2) {
  std::find_first_of(v1.cbegin(), v2.cend(), v2.cbegin(), v2.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
  std::find_first_of(v1.cbegin(), v1.cend(), v2.cbegin(), v1.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
}

void bad_move_find1(std::vector<int> &v1, std::vector<int> &v2, int n) {
  auto i0 = v2.cbegin();
  v1 = std::move(v2);
  std::find(i0, v2.cend(), n); // expected-warning{{Iterators of different containers used where the same container is expected}}
                               // expected-warning@-1{{Method called on moved-from object of type 'std::vector'}}
}

void bad_insert_find(std::vector<int> &v1, std::vector<int> &v2, int n, int m) {
  auto i = std::find(v1.cbegin(), v1.cend(), n);
  v2.insert(i, m); // expected-warning{{Container accessed using foreign iterator argument}}
}

void good_overwrite(std::vector<int> &v1, std::vector<int> &v2, int n) {
  auto i = v1.cbegin();
  i = v2.cbegin();
  v2.insert(i, n); // no-warning
}

void bad_overwrite(std::vector<int> &v1, std::vector<int> &v2, int n) {
  auto i = v1.cbegin();
  i = v2.cbegin();
  v1.insert(i, n); // expected-warning{{Container accessed using foreign iterator argument}}
}

template<typename Container, typename Iterator>
bool is_cend(Container cont, Iterator it) {
  return it == cont.cend();
}

void good_empty(std::vector<int> &v) {
  is_cend(v, v.cbegin()); // no-warning
}

void bad_empty(std::vector<int> &v1, std::vector<int> &v2) {
  is_cend(v1, v2.cbegin()); // expected-warning@-8{{Iterators of different containers used where the same container is expected}}
}

void good_move(std::vector<int> &v1, std::vector<int> &v2) {
  const auto i0 = ++v2.cbegin();
  v1 = std::move(v2);
  v1.erase(i0); // no-warning
}

void bad_move(std::vector<int> &v1, std::vector<int> &v2) {
  const auto i0 = ++v2.cbegin();
  v1 = std::move(v2);
  v2.erase(i0); // expected-warning{{Container accessed using foreign iterator argument}}
                // expected-warning@-1{{Method called on moved-from object of type 'std::vector'}}
}

void bad_move_find2(std::vector<int> &v1, std::vector<int> &v2, int n) {
  auto i0 = --v2.cend();
  v1 = std::move(v2);
  std::find(i0, v2.cend(), n); // expected-warning{{Iterators of different containers used where the same container is expected}}
                               // expected-warning@-1{{Method called on moved-from object of type 'std::vector'}}
}

void bad_move_find3(std::vector<int> &v1, std::vector<int> &v2, int n) {
  auto i0 = v2.cend();
  v1 = std::move(v2);
  std::find(v1.cbegin(), i0, n); // expected-warning{{Iterators of different containers used where the same container is expected}}
}

void bad_comparison(std::vector<int> &v1, std::vector<int> &v2) {
  if (v1.cbegin() != v2.cend()) { // expected-warning{{Iterators of different containers used where the same container is expected}}
    *v1.cbegin();
  }
}

std::vector<int> &return_vector_ref();

void ignore_conjured1() {
  std::vector<int> &v1 = return_vector_ref(), &v2 = return_vector_ref();

  v2.erase(v1.cbegin()); // no-warning
}

void ignore_conjured2() {
  std::vector<int> &v1 = return_vector_ref(), &v2 = return_vector_ref();

  if (v1.cbegin() == v2.cbegin()) {} //no-warning
}
