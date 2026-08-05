// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.NoUncountedMemberChecker \
// RUN:   -analyzer-output=html -o %t %s
// RUN: ls %t | grep report | count 2

// Two instance variables with identical spelling in different
// @interfaces must not collide in the HTML issue hash: the enclosing
// interface differs.

#include "mock-types.h"

__attribute__((objc_root_class))
@interface NSObject
+ (instancetype)alloc;
- (instancetype)init;
@end

@interface FirstClass : NSObject {
  RefCountable* _uncounted;
}
@end

@implementation FirstClass
@end

@interface SecondClass : NSObject {
  RefCountable* _uncounted;
}
@end

@implementation SecondClass
@end
