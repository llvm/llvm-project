// RUN: %clang_cc1 -verify %s

@interface A
@end

template<typename T>
struct X0 {
  void f(T); // expected-error{{interface type 'A' cannot be passed by value}}
};

X0<A> x0a; // expected-note{{instantiation}}


struct test2 { virtual void foo() = 0; }; // expected-note {{unimplemented}}
@interface Test2
- (void) foo: (test2) foo; // expected-error {{parameter type 'test2' is an abstract class}}
@end
