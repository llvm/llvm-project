// REQUIRES: x86-registered-target
// RUN: rm -rf %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only %s -verify
// RUN: not %clang_cc1 -triple x86_64-apple-darwin10 -objcmt-migrate-literals -objcmt-migrate-subscripting -mt-migrate-directory %t %s -x objective-c
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result

// Make sure there is no crash.

@interface NSObject @end
@interface NSNumber : NSObject
@end

@interface NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithInt:(int)value;
@end

void foo(int x = undeclared) { // expected-error {{undeclared}}
  NSNumber *n = [NSNumber numberWithInt:1];
}
