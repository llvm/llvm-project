// RUN: rm -rf %t
// RUN: mkdir -p %t/include %t/subdir
// RUN: cp %S/Inputs/identifier-naming-macro-argument/.clang-tidy %t/.clang-tidy
// RUN: cp %S/Inputs/identifier-naming-macro-argument/include/macro.h %t/include/macro.h
// RUN: cp %S/Inputs/identifier-naming-macro-argument/subdir/.clang-tidy %t/subdir/.clang-tidy
// RUN: cp %s %t/subdir/test.cpp
// RUN: clang-tidy %t/subdir/test.cpp \
// RUN:   --checks=-*,readability-identifier-naming \
// RUN:   -- -I%t/include -std=c++17 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-MESSAGES \
// RUN:       --implicit-check-not="{{warning|error}}:"

#include "macro.h"

int goodFunction(int goodParam) {
  int goodVariable = goodParam;
  return goodVariable;
}

void callWrappedLambda() {
  WRAP([](int wrappedParam) {
    return wrappedParam;
  }(1));
}

// CHECK-MESSAGES: :[[@LINE+2]]:5: warning: invalid case style for function 'BadFunction'
// CHECK-MESSAGES: :[[@LINE+1]]:21: warning: invalid case style for parameter 'BadParam'
int BadFunction(int BadParam) {
  return BadParam;
}
