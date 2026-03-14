// RUN: %clang_cc1 -verify %s

#define NODEREF __attribute__((noderef))

@interface NSObject
+ (id)new;
@end

void func(void) {
  id NODEREF obj = [NSObject new]; // expected-warning{{'noderef' can only be used on an array or pointer type}}
}
