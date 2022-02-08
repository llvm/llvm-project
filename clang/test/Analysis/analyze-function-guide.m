
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyze-function='-[MyClass messageWithFoo:bar:]' \
// RUN:   -triple x86_64-pc-linux-gnu 2>&1 %s \
// RUN: | FileCheck %s -check-prefix=CHECK-MATCH --allow-empty
//
// Expected empty standard output.
// CHECK-MATCH-NOT: Every top-level function was skipped.

// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyze-function='missing_fn' \
// RUN:   -triple x86_64-pc-linux-gnu 2>&1 %s \
// RUN: | FileCheck %s -check-prefix=CHECK-MISSING
//
// CHECK-MISSING: Every top-level function was skipped.
// CHECK-MISSING: Pass the -analyzer-display-progress for tracking which functions are analyzed.

@interface MyClass
- (int)messageWithFoo:(int)foo bar:(int)bar;
@end

@implementation MyClass
- (int)messageWithFoo:(int)foo bar:(int)bar {
  return foo + bar;
}
@end
