// RUN: rm -rf %t
// RUN: mkdir -p %t/subdir
// RUN: cp %S/Inputs/identifier-naming-macro-argument/.clang-tidy %t/.clang-tidy
// RUN: cp %S/Inputs/identifier-naming-macro-argument/subdir/.clang-tidy %t/subdir/.clang-tidy
// RUN: cp %s %t/subdir/test.cpp
// RUN: clang-tidy %t/subdir/test.cpp \
// RUN:   --checks=-*,readability-identifier-naming 2>&1 | FileCheck %s \
// RUN:   -check-prefix=CHECK-MESSAGES

#define WRAP(E) E

int goodFunction(int goodParam) {
  int goodVariable = goodParam;
  return goodVariable;
}

int wrappedExpression = WRAP(1);

WRAP(int wrappedFunction(int wrappedParam) { return wrappedParam; })

void callWrappedLambda() {
  WRAP([](int wrappedParam) {
    return wrappedParam;
  }(1));
}

int BadFunction(int BadParam) {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for function 'BadFunction'
// CHECK-MESSAGES: :[[@LINE-2]]:21: warning: invalid case style for parameter 'BadParam'
  return BadParam;
}
