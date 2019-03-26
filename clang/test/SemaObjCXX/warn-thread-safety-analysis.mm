// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -Wthread-safety-beta -Wno-objc-root-class %s

#include "thread-safety-analysis.h"

@interface MyInterface {
@private
  Lock lock_;
  int value_;
}

- (void)incrementValue;
- (void)decrementValue;

@end

@implementation MyInterface

- (void)incrementValue {
  AutoLock lock(lock_);
  value_ += 1;
}

- (void)decrementValue {
  lock_.Acquire(); // expected-note{{mutex acquired here}}
  value_ -= 1;
} // expected-warning{{mutex 'self->lock_' is still held at the end of function}}

@end
