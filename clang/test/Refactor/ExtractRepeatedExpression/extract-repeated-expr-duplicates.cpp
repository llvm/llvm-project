
struct AClass {
  int method();
  int method2();
};
struct AWrapperClass {
  AClass &object(int x);
};

void duplicatesWithParens(AWrapperClass &wrapper) {
  wrapper.object(0).method();
// CHECK1: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-1]]:3
  ((wrapper).object((0))).method();
// CHECK2: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-1]]:4
}

// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:11:3-20 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:13:4-25 %s | FileCheck --check-prefix=CHECK2 %s


void noDuplicatesWithParens(AWrapperClass &wrapper) {
  wrapper.object(- 1).method();
#ifndef DUPLICATE
  wrapper.object((- 1)).method();
#else
  (wrapper).object(- 1).method();
#endif
// CHECK3: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-6]]:3

  wrapper.object(1 + 2).method();
#ifndef DUPLICATE
  wrapper.object((1 + 2)).method();
#else
  ((wrapper)).object(1 + 2).method();
#endif
// CHECK4: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-6]]:3

  wrapper.object(true ? 0 : 1).method();
#ifndef DUPLICATE
  wrapper.object((true ? 0 : 1)).method();
#else
  ((wrapper)).object(true ? (0) : (1)).method();
#endif
// CHECK5: Initiated the 'extract-repeated-expr-into-var' action at [[@LINE-6]]:3
}

// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:22:3-22 %s -DDUPLICATE | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:30:3-24 %s -DDUPLICATE | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:38:3-31 %s -DDUPLICATE | FileCheck --check-prefix=CHECK5 %s

// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:22:1-32 -in=%s:30:1-34 -in=%s:38:1-41 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
// CHECK-NO: Failed to initiate the refactoring action!

void noDuplicatesWhenSemanticsChange(AWrapperClass &wrapper) {
  wrapper.object(0).method();
  if (true) {
    AWrapperClass wrapperBase;
    AWrapperClass &wrapper = wrapperBase;
    wrapper.object(0).method();
  }
}

// RUN: not clang-refactor-test initiate -action extract-repeated-expr-into-var -in=%s:55:1-30 in=%s:59:1-32 %s 2>&1 | FileCheck --check-prefix=CHECK-NO %s
