@interface I
- (int)func;
@end

@implementation I
- (int)func:(int *)param {
  return *param;
}
@end

void foo(I *i) {
  int *x = 0;
  [i func:x];
}

// RUN: rm -rf %t.output
// RUN: %clang_analyze_cc1 -analyze -analyzer-checker=core -analyzer-output html -o %t.output -Wno-objc-root-class %s
// RUN: cat %t.output/* | FileCheck %s
// CHECK: var relevant_lines = {"1": {"6": 1, "7": 1, "11": 1, "12": 1, "13": 1}};
