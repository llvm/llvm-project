// RUN: %clang_analyze_cc1 -w -triple i386-apple-darwin10 -fblocks -analyzer-checker=core.builtin,debug.ExprInspection -verify %s

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

void test_2() {
  __block int x;
  ^{
    clang_analyzer_explain(&x); // expected-warning-re{{{{^pointer to block variable 'x'$}}}}
  };
  clang_analyzer_explain(&x); // expected-warning-re{{{{^pointer to block variable 'x'$}}}}
}
