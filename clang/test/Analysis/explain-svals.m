// RUN: %clang_analyze_cc1 -w -triple i386-apple-darwin10 -fblocks -verify %s \
// RUN:   -analyzer-checker=core.builtin \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config display-checker-name=false

#include "Inputs/system-header-simulator-objc.h"

void clang_analyzer_explain(void *);

@interface Object : NSObject {
@public
  Object *x;
}
@end

void test_1(Object *p) {
  clang_analyzer_explain(p); // expected-warning-re{{{{^argument 'p'$}}}}
  clang_analyzer_explain(p->x); // expected-warning-re{{{{^initial value of instance variable 'x' of object at argument 'p'$}}}}
  Object *q = [[Object alloc] init];
  clang_analyzer_explain(q); // expected-warning-re{{{{^symbol of type 'Object \*' conjured at statement '\[\[Object alloc\] init\]'$}}}}
  clang_analyzer_explain(q->x); // expected-warning-re{{{{^initial value of instance variable 'x' of object at symbol of type 'Object \*' conjured at statement '\[\[Object alloc\] init\]'$}}}}
}

void test_2(void) {
  __block int x;
  ^{
    clang_analyzer_explain(&x); // expected-warning-re{{{{^pointer to block variable 'x'$}}}}
  };
  clang_analyzer_explain(&x); // expected-warning-re{{{{^pointer to block variable 'x'$}}}}
}

@interface O
+ (instancetype)top_level_class_method:(int)param;
+ (instancetype)non_top_level_class_method:(int)param;
- top_level_instance_method:(int)param;
- non_top_level_instance_method:(int)param;
@end

@implementation O
+ (instancetype)top_level_class_method:(int)param {
  clang_analyzer_explain(&param); // expected-warning-re{{{{^pointer to parameter 'param'$}}}}
}

+ (instancetype)non_top_level_class_method:(int)param {
  clang_analyzer_explain(&param); // expected-warning-re{{{{^pointer to parameter 'param'$}}}}
}

- top_level_instance_method:(int)param {
  clang_analyzer_explain(&param); // expected-warning-re{{{{^pointer to parameter 'param'$}}}}
}

- non_top_level_instance_method:(int)param {
  clang_analyzer_explain(&param); // expected-warning-re{{{{^pointer to parameter 'param'$}}}}
}
@end

void test_3(int n, int m) {
  O *o = [O non_top_level_class_method:n];
  [o non_top_level_instance_method:m];

  void (^block_top_level)(int) = ^(int param) {
    clang_analyzer_explain(&param); // expected-warning-re{{{{^pointer to parameter 'param'$}}}}
    clang_analyzer_explain(&n);     // expected-warning-re{{{{^pointer to parameter 'n'$}}}}
  };
  void (^block_non_top_level)(int) = ^(int param) {
    clang_analyzer_explain(&param); // expected-warning-re{{{{^pointer to parameter 'param'$}}}}
    clang_analyzer_explain(&n);     // expected-warning-re{{{{^pointer to parameter 'n'$}}}}
  };

  block_non_top_level(n);
}
