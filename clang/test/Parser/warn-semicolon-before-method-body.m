// RUN: %clang_cc1 -fsyntax-only -Wsemicolon-before-method-body -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wsemicolon-before-method-body -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// Allow optional semicolon in objc method definition after method prototype,
// warn about it and suggest a fixit.

@interface NSObject
@end

@interface C : NSObject
- (int)z;
@end

@implementation C
- (int)z; // expected-warning {{semicolon before method body is ignored}}
{
  return 0;
}
@end

// CHECK: fix-it:"{{.*}}":{15:9-15:10}:""

