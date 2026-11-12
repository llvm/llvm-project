// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety -Wthread-safety-beta -Wno-objc-root-class %s

struct __attribute__ ((lockable)) Mutex {};

struct Mutex mu1;

int Foo_fun1(int i) __attribute__ ((exclusive_locks_required((mu1)))) {
  return i;
}

@interface test
@end

@implementation test
- (void) PR19541 {
  Foo_fun1(1); // expected-warning{{calling function 'Foo_fun1' requires holding mutex 'mu1' exclusively}}
}

@end
