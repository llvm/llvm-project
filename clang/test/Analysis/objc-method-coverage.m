// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-stats -fblocks %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-ipa=none -analyzer-stats -fblocks %s 2>&1 | FileCheck %s

@interface I
int f() {
  return 0;
}
@end

@implementation I
+ (void *)ff{
  return (void*)0;  
}
@end

// CHECK: ... Statistics Collected ...
// CHECK: 2 AnalysisConsumer - The # of functions analysed (as top level).