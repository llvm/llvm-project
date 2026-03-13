// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedLambdaCapturesChecker -fobjc-runtime-has-weak -fobjc-weak -fobjc-arc -verify %s
// expected-no-diagnostics

#include "objc-mock-types.h"

void someFunction();
template <typename Callback> void call(Callback callback) {
  someFunction();
  callback();
}

NSString *provideStr();
SomeObj *provideSomeObj();

void foo() {
  __weak NSString *weakStr = provideStr();
  __weak SomeObj *weakObj = provideSomeObj();
  auto lambda = [weakStr, weakObj]() {
    return [weakStr length] + [weakObj value];
  };
  call(lambda);
}
