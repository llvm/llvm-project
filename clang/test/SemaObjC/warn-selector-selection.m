// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface Object 
- (void)foo;
@end

@interface Class1
- (void)setWindow:(Object *)wdw;
@end

void foo(void) {
  Object *obj;
  [obj setWindow:0]; // expected-warning{{'Object' may not respond to 'setWindow:'}}
}
