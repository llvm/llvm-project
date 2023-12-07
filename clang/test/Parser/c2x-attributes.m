// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

enum __attribute__((deprecated)) E1 : int; // ok
enum [[deprecated]] E2 : int;

@interface Base
@end

@interface S : Base
- (void) bar;
@end

@interface T : Base
- (S *) foo;
@end


void f(T *t) {
  [[]][[t foo] bar];
}
