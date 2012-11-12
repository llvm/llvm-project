// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify -Wno-objc-root-class %s
// expected-no-diagnostics

// Test reasoning about static locals in ObjCMethods. 
int *getValidPtr();
@interface Radar11275803
- (int) useStaticInMethod;
@end
@implementation Radar11275803

- (int) useStaticInMethod
{
  static int *explInit = 0;
  static int implInit;
  if (!implInit)
    explInit = getValidPtr();
  return *explInit; //no-warning
}
@end