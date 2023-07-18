// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics
// rdar://13749180

@interface NSDictionary
- (id)objectForKeyedSubscript:(id)key;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
- (int &) random;
@end

@class NSString;

template <class T, class U = T> T tfoo(U x) { return x; }

void func() {
  NSDictionary* foo;
  NSString* result = foo[@"bar"] ? : foo[@"baz"];

  int (*fn)(int) = (&tfoo<int> ?: 0);

  int x = 0;
  const int &y = foo.random ?: x;
}
