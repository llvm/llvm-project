// RUN: %clang_cc1 -fsyntax-only -verify %s

// rdar://15505492
@import Foundation; // expected-error {{use of '@import' when modules are disabled}}

@interface Subclass 
+ (int)magicNumber;
@end

int main (void) {
  return Subclass.magicNumber;
}

