// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface foo
@end

@implementation foo
@end

@interface bar
-(void) my_method:(foo) my_param; // expected-error {{interface type 'foo' cannot be passed by value; did you forget * in 'foo'}}
- (foo)cccccc:(long)ddddd;  // expected-error {{interface type 'foo' cannot be returned by value; did you forget * in 'foo'}}
@end

@implementation bar
-(void) my_method:(foo) my_param  // expected-error {{interface type 'foo' cannot be passed by value; did you forget * in 'foo'}}
{
}
- (foo)cccccc:(long)ddddd // expected-error {{interface type 'foo' cannot be returned by value; did you forget * in 'foo'}}
{
}
@end

void somefunc(foo x) {} // expected-error {{interface type 'foo' cannot be passed by value; did you forget * in 'foo'}}
foo somefunc2() {} // expected-error {{interface type 'foo' cannot be returned by value; did you forget * in 'foo'}}

// rdar://6780761
void f0(foo *a0) {
  extern void g0(int x, ...);
  g0(1, *(foo*)a0);  // expected-error {{cannot pass object with interface type 'foo' by value through variadic function}}
}

// rdar://8421082
enum bogus; // expected-note {{forward declaration of 'enum bogus'}}

@interface fee  {
}
- (void)crashMe:(enum bogus)p;
@end

@implementation fee
- (void)crashMe:(enum bogus)p { // expected-error {{variable has incomplete type 'enum bogus'}}
}
@end

