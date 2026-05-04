// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,debug.ExprInspection -verify %s

// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix,alpha.unix,debug.ExprInspection -verify %s
//
// expected-no-diagnostics

#include "Inputs/system-header-simulator-cxx.h"

void test_std_getline() {
  std::string userid, comment;
  // MallocChecker should not confuse the POSIX function getline() and the
  // unrelated C++ standard library function std::getline.
  std::getline(std::cin, userid, ' '); // no-crash
  std::getline(std::cin, comment); // no-crash
}
