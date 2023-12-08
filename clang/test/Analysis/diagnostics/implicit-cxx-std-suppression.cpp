// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete,debug.ExprInspection -analyzer-config c++-container-inlining=true -analyzer-config c++-stdlib-inlining=false -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete,debug.ExprInspection -analyzer-config c++-container-inlining=true -analyzer-config c++-stdlib-inlining=true -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete,debug.ExprInspection -analyzer-config c++-container-inlining=true -analyzer-config c++-stdlib-inlining=false -std=c++11 -DTEST_INLINABLE_ALLOCATORS -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete,debug.ExprInspection -analyzer-config c++-container-inlining=true -analyzer-config c++-stdlib-inlining=true -std=c++11 -DTEST_INLINABLE_ALLOCATORS -verify %s

// expected-no-diagnostics

#include "../Inputs/system-header-simulator-cxx-std-suppression.h"

void testList_pop_front(std::list<int> list) {
  while(!list.empty())
    list.pop_front();  // no-warning
}

void testBasicStringSuppression() {
  std::basic_string<uint8_t> v;
  v.push_back(1); // no-warning
}

void testBasicStringSuppression_append() {
  std::basic_string<char32_t> v;
  v += 'c'; // no-warning
}

void testBasicStringSuppression_assign(std::basic_string<char32_t> &v,
                                       const std::basic_string<char32_t> &v2) {
  v = v2; // no-warning
}

class MyEngine;
void testSuppression_independent_bits_engine(MyEngine& e) {
  std::__independent_bits_engine<MyEngine, unsigned int> x(e, 64); // no-warning
}

void testSuppression_std_shared_pointer() {
  std::shared_ptr<int> p(new int(1));

  p = nullptr; // no-warning
}
