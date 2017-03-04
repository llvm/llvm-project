// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorPastEnd -analyzer-eagerly-assume -analyzer-config c++-container-inlining=false %s -verify
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,alpha.cplusplus.IteratorPastEnd -analyzer-eagerly-assume -analyzer-config c++-container-inlining=true -DINLINE=1 %s -verify

#include "Inputs/system-header-simulator-cxx.h"

void simple_good(const std::vector<int> &v) {
  auto i = v.end();
  if (i != v.end())
    *i; // no-warning
}

void simple_good_negated(const std::vector<int> &v) {
  auto i = v.end();
  if (!(i == v.end()))
    *i; // no-warning
}

void simple_bad(const std::vector<int> &v) {
  auto i = v.end();
  *i; // expected-warning{{Iterator accessed past its end}}
}

void copy(const std::vector<int> &v) {
  auto i1 = v.end();
  auto i2 = i1;
  *i2; // expected-warning{{Iterator accessed past its end}}
}

void decrease(const std::vector<int> &v) {
  auto i = v.end();
  --i;
  *i; // no-warning
}

void copy_and_decrease1(const std::vector<int> &v) {
  auto i1 = v.end();
  auto i2 = i1;
  --i1;
  *i1; // no-warning
}

void copy_and_decrease2(const std::vector<int> &v) {
  auto i1 = v.end();
  auto i2 = i1;
  --i1;
  *i2; // expected-warning{{Iterator accessed past its end}}
}

void copy_and_increase1(const std::vector<int> &v) {
  auto i1 = v.begin();
  auto i2 = i1;
  ++i1;
  if (i1 == v.end())
    *i2; // no-warning
}

void copy_and_increase2(const std::vector<int> &v) {
  auto i1 = v.begin();
  auto i2 = i1;
  ++i1;
  if (i2 == v.end())
    *i2; // expected-warning{{Iterator accessed past its end}}
}

void good_find(std::vector<int> &vec, int e) {
  auto first = std::find(vec.begin(), vec.end(), e);
  if (vec.end() != first)
    *first; // no-warning
}

void bad_find(std::vector<int> &vec, int e) {
  auto first = std::find(vec.begin(), vec.end(), e);
  *first; // expected-warning{{Iterator accessed past its end}}
}

void good_find_end(std::vector<int> &vec, std::vector<int> &seq) {
  auto last = std::find_end(vec.begin(), vec.end(), seq.begin(), seq.end());
  if (vec.end() != last)
    *last; // no-warning
}

void bad_find_end(std::vector<int> &vec, std::vector<int> &seq) {
  auto last = std::find_end(vec.begin(), vec.end(), seq.begin(), seq.end());
  *last; // expected-warning{{Iterator accessed past its end}}
}

void good_find_first_of(std::vector<int> &vec, std::vector<int> &seq) {
  auto first =
      std::find_first_of(vec.begin(), vec.end(), seq.begin(), seq.end());
  if (vec.end() != first)
    *first; // no-warning
}

void bad_find_first_of(std::vector<int> &vec, std::vector<int> &seq) {
  auto first = std::find_end(vec.begin(), vec.end(), seq.begin(), seq.end());
  *first; // expected-warning{{Iterator accessed past its end}}
}

bool odd(int i) { return i % 2; }

void good_find_if(std::vector<int> &vec) {
  auto first = std::find_if(vec.begin(), vec.end(), odd);
  if (vec.end() != first)
    *first; // no-warning
}

void bad_find_if(std::vector<int> &vec, int e) {
  auto first = std::find_if(vec.begin(), vec.end(), odd);
  *first; // expected-warning{{Iterator accessed past its end}}
}

void good_find_if_not(std::vector<int> &vec) {
  auto first = std::find_if_not(vec.begin(), vec.end(), odd);
  if (vec.end() != first)
    *first; // no-warning
}

void bad_find_if_not(std::vector<int> &vec, int e) {
  auto first = std::find_if_not(vec.begin(), vec.end(), odd);
  *first; // expected-warning{{Iterator accessed past its end}}
}

void good_lower_bound(std::vector<int> &vec, int e) {
  auto first = std::lower_bound(vec.begin(), vec.end(), e);
  if (vec.end() != first)
    *first; // no-warning
}

void bad_lower_bound(std::vector<int> &vec, int e) {
  auto first = std::lower_bound(vec.begin(), vec.end(), e);
  *first; // expected-warning{{Iterator accessed past its end}}
}

void good_upper_bound(std::vector<int> &vec, int e) {
  auto last = std::lower_bound(vec.begin(), vec.end(), e);
  if (vec.end() != last)
    *last; // no-warning
}

void bad_upper_bound(std::vector<int> &vec, int e) {
  auto last = std::lower_bound(vec.begin(), vec.end(), e);
  *last; // expected-warning{{Iterator accessed past its end}}
}

void good_search(std::vector<int> &vec, std::vector<int> &seq) {
  auto first = std::search(vec.begin(), vec.end(), seq.begin(), seq.end());
  if (vec.end() != first)
    *first; // no-warning
}

void bad_search(std::vector<int> &vec, std::vector<int> &seq) {
  auto first = std::search(vec.begin(), vec.end(), seq.begin(), seq.end());
  *first; // expected-warning{{Iterator accessed past its end}}
}

void good_search_n(std::vector<int> &vec, std::vector<int> &seq) {
  auto nth = std::search_n(vec.begin(), vec.end(), seq.begin(), seq.end());
  if (vec.end() != nth)
    *nth; // no-warning
}

void bad_search_n(std::vector<int> &vec, std::vector<int> &seq) {
  auto nth = std::search_n(vec.begin(), vec.end(), seq.begin(), seq.end());
  *nth; // expected-warning{{Iterator accessed past its end}}
}

template <class InputIterator, class T>
InputIterator nonStdFind(InputIterator first, InputIterator last,
                         const T &val) {
  for (auto i = first; i != last; ++i) {
    if (*i == val) {
      return i;
    }
  }
  return last;
}

void good_non_std_find(std::vector<int> &vec, int e) {
  auto first = nonStdFind(vec.begin(), vec.end(), e);
  if (vec.end() != first)
    *first; // no-warning
}

void bad_non_std_find(std::vector<int> &vec, int e) {
  auto first = nonStdFind(vec.begin(), vec.end(), e);
  *first; // expected-warning{{Iterator accessed past its end}}
}

void tricky(std::vector<int> &vec, int e) {
  const auto first = vec.begin();
  const auto comp1 = (first != vec.end()), comp2 = (first == vec.end());
  if (comp1)
    *first;
}

void loop(std::vector<int> &vec, int e) {
  auto start = vec.begin();
  while (true) {
    auto item = std::find(start, vec.end(), e);
    if (item == vec.end())
      break;
    *item;          // no-warning
    start = ++item; // no-warning
  }
}
