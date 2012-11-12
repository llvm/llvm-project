// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface Subclass 
+ (int)magicNumber;
@end

int main (void) {
  return Subclass.magicNumber;
}

