// RUN: %clang_cc1 -std=gnu++98 -fsyntax-only -verify -Wno-objc-root-class %s
struct Y { 
  Y(); 

private:
  ~Y(); // expected-note 3{{declared private here}}
};

template<typename T>
struct X : T { }; // expected-error 2{{private destructor}}

struct Z; // expected-note{{forward declaration}}

@interface A {
  X<Y> x; // expected-note{{implicit destructor}}
  Y y; // expected-error{{private destructor}}
}
@end

@implementation A // expected-note{{implicit default constructor}}
@end

@interface B {
  Z z; // expected-error{{incomplete type}}
}
@end

@implementation B
@end

// <rdar://problem/11284902>
template<typename T> struct Incomplete; // expected-note{{declared here}}

@interface C {
  Incomplete<int> a[4][4][4]; // expected-error{{implicit instantiation of undefined template 'Incomplete<int>'}}
}
@end
