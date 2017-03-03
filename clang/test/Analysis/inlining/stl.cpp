// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete,debug.ExprInspection -analyzer-config c++-container-inlining=true -analyzer-config c++-stdlib-inlining=false -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete,debug.ExprInspection -analyzer-config c++-container-inlining=true -analyzer-config c++-stdlib-inlining=true -std=c++11 -DINLINE=1 -verify %s

#include "../Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

void testVector(std::vector<int> &nums) {
  if (nums.begin() != nums.end()) return;
  
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
