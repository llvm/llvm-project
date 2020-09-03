// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.MismatchedIterator -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=false %s -verify
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.MismatchedIterator -analyzer-config aggressive-binary-operation-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

#include "Inputs/system-header-simulator-cxx.h"

void good_insert1(std::vector<int> &V, int n) {
  V.insert(V.cbegin(), n); // no-warning
}

void good_insert2(std::vector<int> &V, int len, int n) {
  V.insert(V.cbegin(), len, n); // no-warning
}

void good_insert3(std::vector<int> &V1, std::vector<int> &V2) {
  V1.insert(V1.cbegin(), V2.cbegin(), V2.cend()); // no-warning
}

void good_insert4(std::vector<int> &V, int len, int n) {
  V.insert(V.cbegin(), {n-1, n, n+1}); // no-warning
}

void good_insert_find(std::vector<int> &V, int n, int m) {
  auto i = std::find(V.cbegin(), V.cend(), n);
  V.insert(i, m); // no-warning
}

void good_erase1(std::vector<int> &V) {
  V.erase(V.cbegin()); // no-warning
}

void good_erase2(std::vector<int> &V) {
  V.erase(V.cbegin(), V.cend()); // no-warning
}

void good_emplace(std::vector<int> &V, int n) {
  V.emplace(V.cbegin(), n); // no-warning
}

void good_ctor(std::vector<int> &V) {
  std::vector<int> new_v(V.cbegin(), V.cend()); // no-warning
}

void good_find(std::vector<int> &V, int n) {
  std::find(V.cbegin(), V.cend(), n); // no-warning
}

void good_find_first_of(std::vector<int> &V1, std::vector<int> &V2) {
  std::find_first_of(V1.cbegin(), V1.cend(), V2.cbegin(), V2.cend()); // no-warning
}

void good_copy(std::vector<int> &V1, std::vector<int> &V2, int n) {
  std::copy(V1.cbegin(), V1.cend(), V2.begin()); // no-warning
}

void bad_insert1(std::vector<int> &V1, std::vector<int> &V2, int n) {
  V2.insert(V1.cbegin(), n); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_insert2(std::vector<int> &V1, std::vector<int> &V2, int len, int n) {
  V2.insert(V1.cbegin(), len, n); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_insert3(std::vector<int> &V1, std::vector<int> &V2) {
  V2.insert(V1.cbegin(), V2.cbegin(), V2.cend()); // expected-warning{{Container accessed using foreign iterator argument}}
  V1.insert(V1.cbegin(), V1.cbegin(), V2.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
  V1.insert(V1.cbegin(), V2.cbegin(), V1.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
}

void bad_insert4(std::vector<int> &V1, std::vector<int> &V2, int len, int n) {
  V2.insert(V1.cbegin(), {n-1, n, n+1}); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_erase1(std::vector<int> &V1, std::vector<int> &V2) {
  V2.erase(V1.cbegin()); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_erase2(std::vector<int> &V1, std::vector<int> &V2) {
  V2.erase(V2.cbegin(), V1.cend()); // expected-warning{{Container accessed using foreign iterator argument}}
  V2.erase(V1.cbegin(), V2.cend()); // expected-warning{{Container accessed using foreign iterator argument}}
  V2.erase(V1.cbegin(), V1.cend()); // expected-warning{{Container accessed using foreign iterator argument}}
}

void bad_emplace(std::vector<int> &V1, std::vector<int> &V2, int n) {
  V2.emplace(V1.cbegin(), n); // expected-warning{{Container accessed using foreign iterator argument}}
}

void good_comparison(std::vector<int> &V) {
  if (V.cbegin() == V.cend()) {} // no-warning
}

void bad_comparison(std::vector<int> &V1, std::vector<int> &V2) {
  if (V1.cbegin() != V2.cend()) {} // expected-warning{{Iterators of different containers used where the same container is expected}}
}

void bad_ctor(std::vector<int> &V1, std::vector<int> &V2) {
  std::vector<int> new_v(V1.cbegin(), V2.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
}

void bad_find(std::vector<int> &V1, std::vector<int> &V2, int n) {
  std::find(V1.cbegin(), V2.cend(), n); // expected-warning{{Iterators of different containers used where the same container is expected}}
}

void bad_find_first_of(std::vector<int> &V1, std::vector<int> &V2) {
  std::find_first_of(V1.cbegin(), V2.cend(), V2.cbegin(), V2.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
  std::find_first_of(V1.cbegin(), V1.cend(), V2.cbegin(), V1.cend()); // expected-warning{{Iterators of different containers used where the same container is expected}}
}

std::vector<int> &return_vector_ref();

void ignore_conjured1() {
  std::vector<int> &V1 = return_vector_ref(), &V2 = return_vector_ref();

  V2.erase(V1.cbegin()); // no-warning
}

void ignore_conjured2() {
  std::vector<int> &V1 = return_vector_ref(), &V2 = return_vector_ref();

  if (V1.cbegin() == V2.cbegin()) {} //no-warning
}

template<typename T>
struct cont_with_ptr_iterator {
  T *begin() const;
  T *end() const;
};

void comparison_ptr_iterator(cont_with_ptr_iterator<int> &C1,
                             cont_with_ptr_iterator<int> &C2) {
  if (C1.begin() != C2.end()) {} // expected-warning{{Iterators of different containers used where the same container is expected}}
}

