// RUN: %clang_analyze_cc1 -std=c++11\
// RUN: -analyzer-checker=core,cplusplus,debug.DebugIteratorModeling,debug.ExprInspection\
// RUN: %s 2>&1 | FileCheck %s

// XFAIL: *

// CHECK: checker cannot be enabled with analyzer option 'aggressive-binary-operation-simplification' == false

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

void comparison(std::vector<int> &V) {
  clang_analyzer_eval(V.begin() == V.end()); // no-crash
}
