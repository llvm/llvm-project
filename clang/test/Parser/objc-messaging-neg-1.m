// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface A
+(void) foo:(int) a;
@end

int main(void) {
  id a;
  [a bla:0 6:7]; // expected-error {{expected ']'}}
  [A foo bar]; // expected-error {{expected ':'}}
  [A foo bar bar1]; // expected-error {{expected ':'}}
  [] {}; // expected-error {{expected expression}}
}
