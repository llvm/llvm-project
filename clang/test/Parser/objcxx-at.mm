// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface B {
  int i;
}
@end

struct Z {
  @defs(B); // expected-error{{@defs is not supported in Objective-C++}}
};

struct Y { // expected-note{{to match this '{'}}
  struct X { } // expected-error{{expected ';' after struct}}
    @interface A // expected-error{{unexpected '@' in member specification}}
} // expected-error{{expected '}'}} expected-error{{expected ';' after struct}}
