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

// Ensure that this function is properly marked as a failure.
void func_with_bad_call(bar* b, foo* f) {
  [b cccccc:5]; // expected-warning {{instance method '-cccccc:' not found}}
                // expected-note@-17 {{receiver is instance of class declared here}}
}

void somefunc(foo x) {} // expected-error {{interface type 'foo' cannot be passed by value; did you forget * in 'foo'}}
foo somefunc2(void) {} // expected-error {{interface type 'foo' cannot be returned by value; did you forget * in 'foo'}}

void f0(foo *a0) {
  extern void g0(int x, ...);
  g0(1, *(foo*)a0);  // expected-error {{cannot pass object with interface type 'foo' by value through variadic function}}
}

enum bogus; // expected-note {{forward declaration of 'enum bogus'}}

@interface fee  {
}
- (void)crashMe:(enum bogus)p;
@end

@implementation fee
- (void)crashMe:(enum bogus)p { // expected-error {{variable has incomplete type 'enum bogus'}}
}
@end

@interface arrayfun
- (int[6])arrayRet; // expected-error {{function cannot return array type 'int[6]'}}
- (int(void))funcRet; // expected-error {{function cannot return function type 'int (void)'}}
@end

@interface qux
- (void) my_method: (int)arg; // expected-note {{method 'my_method:' declared here}}
@end

// FIXME: The diagnostic and recovery here could probably be improved.
@implementation qux // expected-warning {{method definition for 'my_method:' not found}}
- (void) my_method: (int) { // expected-error {{expected identifier}}
}
@end
