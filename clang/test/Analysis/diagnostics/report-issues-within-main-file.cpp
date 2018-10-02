// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -analyzer-output=plist-multi-file -analyzer-config report-in-main-source-file=true %s -o %t.plist
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/report-issues-within-main-file.cpp.plist
#include "Inputs/include/report-issues-within-main-file.h"

void mainPlusHeader() {
  auto_ptr<int> B (new int[5]);
}

void auxInMain() {
  int j = 0;
  j++;
  cause_div_by_zero_in_header(j);
  j--;
}
void mainPlusMainPlusHeader() {
  int i = 0;
  i++;
  auxInMain();
  i++;
}

void causeDivByZeroInMain(int in) {
  int m = 0;
  m = in/m;
  m++;
}
void mainPlusMain() {
  int i = 0;
  i++;
  causeDivByZeroInMain(i);
  i++;
}

void causeDivByZeroInMain2(int in) {
  int m2 = 0;
  m2 = in/m2;
  m2++;
}

void mainPlustHeaderCallAndReturnPlusMain() {
  int i = 0;
  i++;
  do_something(i);
  causeDivByZeroInMain2(i);
  i++;
}

void callInMacro() {
  int j = 0;
  j++;
  CALLS_BUGGY_FUNCTION2;
  j--;
}

void callInMacro3() {
  int j = 0;
  j++;
  CALLS_BUGGY_FUNCTION3;
  j--;
}

void callCallInMacro3() {
  callInMacro3();
}

void callInMacroArg() {
  int j = 0;
  j++;
  TAKE_CALL_AS_ARG(cause_div_by_zero_in_header4(5));
  j--;
}
