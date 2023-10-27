// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

__attribute__((objc_root_class))
@interface Root {
  Class isa;
}
@end

@interface A
@property (strong) id x;
@end

void test0(A *a) {
  int kind = _Generic(a.x, id : 0, int : 1, float : 2);
}
