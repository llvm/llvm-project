// RUN: %clang_cc1 -Wconversion -fsyntax-only %s -verify

typedef signed char BOOL;
__attribute__((objc_root_class)) @interface RDar14415662
@property (readonly) BOOL stuff;
@property (readwrite) BOOL otherStuff;
@end

void radar14415662(RDar14415662 *f, char x, int y) {
  f.otherStuff = !f.stuff; // no-warning
  BOOL b = !f.stuff; // no-warning

  // True positive to verify warning is working.
  x = y; // expected-warning {{implicit conversion loses integer precision: 'int' to 'char'}}
}

__attribute__((objc_root_class)) @interface DoubleProp
@property double d;
@end

void use_double_prop(DoubleProp *dp) {
  double local = 42;
  dp.d += local; // no warning
}
