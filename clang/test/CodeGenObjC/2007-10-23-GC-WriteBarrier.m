// RUN: %clang_cc1 -emit-llvm %s -o /dev/null -fobjc-gc

typedef unsigned int NSUInteger;
__attribute__((objc_gc(strong))) float *_scores;

void foo(int i, float f) {
  _scores[i] = f; 
}
