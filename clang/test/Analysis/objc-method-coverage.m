// REQUIRES: asserts
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-stats -fblocks %s 2>&1 | FileCheck %s
@interface I
int f(void) {
  return 0;
}
@end

@implementation I
+ (void *)ff{
  return (void*)0;  
}
@end

// CHECK: ... Statistics Collected ...
// CHECK: 2 AnalysisConsumer - The # of functions and blocks analyzed (as top level with inlining turned on).
// CHECK: 100 AnalysisConsumer - The % of reachable basic blocks.
