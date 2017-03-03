// RUN: %check_clang_tidy %s google-readability-casting %t -- -- -x c
// The testing script always adds .cpp extension to the input file name, so we
// need to run clang-tidy directly in order to verify handling of .c files:
// RUN: clang-tidy --checks=-*,google-readability-casting %s -- -x c++ | FileCheck %s -check-prefix=CHECK-MESSAGES -implicit-check-not='{{warning|error}}:'
// RUN: cp %s %t.main_file.cpp
// RUN: clang-tidy --checks=-*,google-readability-casting -header-filter='.*' %t.main_file.cpp -- -I%S -DTEST_INCLUDE -x c++ | FileCheck %s -check-prefix=CHECK-MESSAGES -implicit-check-not='{{warning|error}}:'

#ifdef TEST_INCLUDE

#undef TEST_INCLUDE
#include "google-readability-casting.c"

#else

void f(const char *cpc) {
  const char *cpc2 = (const char*)cpc;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: redundant cast to the same type [google-readability-casting]
  // CHECK-FIXES: const char *cpc2 = cpc;
  char *pc = (char*)cpc;
  typedef const char *Typedef1;
  (Typedef1)cpc;
}

#endif
