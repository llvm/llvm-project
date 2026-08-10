// RUN: %clang_cc1 -fsyntax-only -Wthread-safety -Wno-objc-root-class %s

// Thread safety analysis used to crash when used with ObjC methods.

#include "thread-safety-analysis.h"

@interface MyInterface
- (void)doIt:(class Lock *)myLock;
@end

@implementation MyInterface
- (void)doIt:(class Lock *)myLock {
  AutoLock lock(*myLock);
}
@end
