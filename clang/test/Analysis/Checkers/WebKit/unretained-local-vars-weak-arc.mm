// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedLocalVarsChecker -fobjc-runtime-has-weak -fobjc-weak -fobjc-arc -verify %s
// expected-no-diagnostics

#include "objc-mock-types.h"

NSString *provideStr();
SomeObj *provideSomeObj();

int foo() {
  __weak NSString *weakStr = provideStr();
  __weak SomeObj *weakObj = provideSomeObj();
  return [weakStr length] + [weakObj value];
}
