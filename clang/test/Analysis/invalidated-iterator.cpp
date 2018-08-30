// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.InvalidatedIterator -analyzer-config aggressive-relational-comparison-simplification=true -analyzer-config c++-container-inlining=false %s -verify
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.InvalidatedIterator -analyzer-config aggressive-relational-comparison-simplification=true -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

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
