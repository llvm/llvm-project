// RUN: %clang_cc1 -fsyntax-only -verify %s

@import Foundation; // expected-error {{use of '@import' when modules are disabled}}

@interface Subclass 
+ (int)magicNumber;
@end

int main (void) {
  return Subclass.magicNumber;
}

