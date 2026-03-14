// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface B1 {
@public
  double fill_B;	// expected-note {{previous declaration is here}}
  unsigned : 0;
}
@end

@interface B : B1 {
@public
  int one;	// expected-note {{previous declaration is here}}
  int one;	// expected-error {{duplicate member 'one'}}
  unsigned : 0;
}
@end

@interface A : B {
@public
  int fill_B;	// expected-error {{duplicate member 'fill_B'}}
}
@end
