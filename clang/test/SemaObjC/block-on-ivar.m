// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

@interface X {
  // expected-warning@-1 {{class 'X' defined without specifying a base class}}
  // expected-note@-2 {{add a super class to fix this problem}}
  __block int x; // expected-error {{'__block' is not allowed on an Objective-C instance variable}}
}
@end

@implementation X
@end
