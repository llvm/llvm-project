// REQUIRES: x86-registered-target
// RUN: rm -rf %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -x objective-c %S/Common.h -emit-pch -o %t.pch
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -objcmt-migrate-literals -objcmt-migrate-subscripting -mt-migrate-directory %t %s -x objective-c -include-pch %t.pch
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s.result -include-pch %t.pch

@interface NSNumber : NSObject
@end

@interface NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithInt:(int)value;
@end

void foo(void) {
  NSNumber *n = [NSNumber numberWithInt:1];
}
