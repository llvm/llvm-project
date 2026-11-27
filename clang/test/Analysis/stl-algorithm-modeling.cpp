// RUN: %clang_analyze_cc1 -std=c++17 %s\
// RUN:  -analyzer-checker=core,cplusplus,alpha.cplusplus.STLAlgorithmModeling,debug.DebugIteratorModeling,debug.ExprInspection\
// RUN:  -analyzer-config aggressive-binary-operation-simplification=true\
// RUN:  -verify

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

template <typename Iterator>
long clang_analyzer_iterator_position(const Iterator&);

template <typename Iter> Iter return_any_iterator(const Iter &It);

void test_find1(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::find(i1, i2, n);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find2(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::find(std::execution::sequenced_policy(), i1, i2, n);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

bool odd(int i) { return i % 2; }

void test_find_if1(std::vector<int> V) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::find_if(i1, i2, odd);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_if2(std::vector<int> V) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::find_if(std::execution::sequenced_policy(), i1, i2, odd);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_if_not1(std::vector<int> V) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::find_if_not(i1, i2, odd);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_if_not2(std::vector<int> V) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::find_if_not(std::execution::sequenced_policy(), i1, i2,
                                   odd);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_first_of1(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::find_first_of(i1, i2, i3, i4);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_first_of2(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::find_first_of(std::execution::sequenced_policy(),
                                     i1, i2, i3, i4);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_first_of3(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::find_first_of(i1, i2, i3, i4, odd);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_first_of4(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::find_first_of(std::execution::sequenced_policy(),
                                     i1, i2, i3, i4, odd);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_end1(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::find_end(i1, i2, i3, i4);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_end2(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::find_end(std::execution::sequenced_policy(),
                                i1, i2, i3, i4);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_end3(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::find_end(i1, i2, i3, i4, odd);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_find_end4(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::find_end(std::execution::sequenced_policy(),
                                i1, i2, i3, i4, odd);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

bool compare(int, int);

void test_lower_bound1(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::lower_bound(i1, i2, n);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_lower_bound2(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::lower_bound(i1, i2, n, compare);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_upper_bound1(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::upper_bound(i1, i2, n);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_upper_bound2(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::upper_bound(i1, i2, n, compare);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_search1(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::search(i1, i2, i3, i4);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_search2(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::search(std::execution::sequenced_policy(),
                              i1, i2, i3, i4);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_search3(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::search(i1, i2, i3, i4, odd);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_search4(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::search(std::execution::sequenced_policy(),
                              i1, i2, i3, i4, odd);

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_search5(std::vector<int> V1, std::vector<int> V2) {
  const auto i1 = return_any_iterator(V1.begin());
  const auto i2 = return_any_iterator(V1.begin());
  const auto i3 = return_any_iterator(V2.begin());
  const auto i4 = return_any_iterator(V2.begin());

  const auto i5 = std::search(i1, i2, std::default_searcher(i3, i4));

  clang_analyzer_eval(i5 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i5) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i5) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_search_n1(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::search_n(i1, i2, 2, n);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_search_n2(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::search_n(std::execution::sequenced_policy(),
                                i1, i2, 2, n);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_search_n3(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::search_n(i1, i2, 2, n, compare);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}

void test_search_n4(std::vector<int> V, int n) {
  const auto i1 = return_any_iterator(V.begin());
  const auto i2 = return_any_iterator(V.begin());

  const auto i3 = std::search_n(std::execution::sequenced_policy(),
                                i1, i2, 2, n, compare);

  clang_analyzer_eval(i3 == i2); // expected-warning{{FALSE}}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i1)); // expected-warning@-1{{FALSE}}

  clang_analyzer_eval(clang_analyzer_iterator_position(i3) <
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{TRUE}}
  clang_analyzer_eval(clang_analyzer_iterator_position(i3) >=
                      clang_analyzer_iterator_position(i2)); // expected-warning@-1{{FALSE}}
}
