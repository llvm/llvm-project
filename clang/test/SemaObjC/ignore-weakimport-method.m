// RUN: %clang_cc1  -fsyntax-only -verify %s
// expected-no-diagnostics
@interface foo 
+ (void) cx __attribute__((weak_import));
- (void) x __attribute__((weak_import));
@end

