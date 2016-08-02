// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete,debug.ExprInspection -analyzer-config c++-container-inlining=true -analyzer-config c++-stdlib-inlining=false -std=c++11 -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete,debug.ExprInspection -analyzer-config c++-container-inlining=true -analyzer-config c++-stdlib-inlining=true -std=c++11 -DINLINE=1 -verify %s

#include "../Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

void testVector(std::vector<int> &nums) {
  if (nums.begin()) return;
  if (nums.end()) return;
  
  clang_analyzer_eval(nums.size() == 0);
#if INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif
}

void testException(std::exception e) {
  // Notice that the argument is NOT passed by reference, so we can devirtualize.
  const char *x = e.what();
  clang_analyzer_eval(x == 0);
#if INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif
}

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
  v = v2;
}

class MyEngine;
void testSupprerssion_independent_bits_engine(MyEngine& e) {
  std::__independent_bits_engine<MyEngine, unsigned int> x(e, 64); // no-warning
}
