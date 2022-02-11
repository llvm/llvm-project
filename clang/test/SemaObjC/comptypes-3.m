// RUN: %clang_cc1 -fsyntax-only -verify %s

#define nil (void *)0;

extern void foo(void);

@protocol MyProtocolA
- (void) methodA;
@end

@protocol MyProtocolB
- (void) methodB;
@end

@protocol MyProtocolAB <MyProtocolA, MyProtocolB>
@end

@protocol MyProtocolAC <MyProtocolA>
- (void) methodC;
@end

int main(void)
{
  id<MyProtocolA> obj_a = nil;
  id<MyProtocolB> obj_b = nil;
  id<MyProtocolAB> obj_ab = nil;
  id<MyProtocolAC> obj_ac = nil;

  obj_a = obj_b;  // expected-warning {{assigning to 'id<MyProtocolA>' from incompatible type 'id<MyProtocolB>'}}
  obj_a = obj_ab; /* Ok */
  obj_a = obj_ac; /* Ok */
  
  obj_b = obj_a;  // expected-warning {{assigning to 'id<MyProtocolB>' from incompatible type 'id<MyProtocolA>'}}
  obj_b = obj_ab; /* Ok */
  obj_b = obj_ac; // expected-warning {{assigning to 'id<MyProtocolB>' from incompatible type 'id<MyProtocolAC>'}}
  
  obj_ab = obj_a;  // expected-warning {{assigning to 'id<MyProtocolAB>' from incompatible type 'id<MyProtocolA>'}}
  obj_ab = obj_b;  // expected-warning {{assigning to 'id<MyProtocolAB>' from incompatible type 'id<MyProtocolB>'}}
  obj_ab = obj_ac; // expected-warning {{assigning to 'id<MyProtocolAB>' from incompatible type 'id<MyProtocolAC>'}}
  
  obj_ac = obj_a;  // expected-warning {{assigning to 'id<MyProtocolAC>' from incompatible type 'id<MyProtocolA>'}}
  obj_ac = obj_b;  // expected-warning {{assigning to 'id<MyProtocolAC>' from incompatible type 'id<MyProtocolB>'}}
  obj_ac = obj_ab; // expected-warning {{assigning to 'id<MyProtocolAC>' from incompatible type 'id<MyProtocolAB>'}}

  if (obj_a == obj_b) foo (); // expected-warning {{comparison of distinct pointer types ('id<MyProtocolA>' and 'id<MyProtocolB>')}}
  if (obj_b == obj_a) foo (); // expected-warning {{comparison of distinct pointer types ('id<MyProtocolB>' and 'id<MyProtocolA>')}}

  if (obj_a == obj_ab) foo (); /* Ok */
  if (obj_ab == obj_a) foo (); /* Ok */ 

  if (obj_a == obj_ac) foo (); /* Ok */ 
  if (obj_ac == obj_a) foo (); /* Ok */ 

  if (obj_b == obj_ab) foo (); /* Ok */ 
  if (obj_ab == obj_b) foo (); /* Ok */ 

  if (obj_b == obj_ac) foo (); // expected-warning {{comparison of distinct pointer types ('id<MyProtocolB>' and 'id<MyProtocolAC>')}} 
  if (obj_ac == obj_b) foo (); // expected-warning {{comparison of distinct pointer types ('id<MyProtocolAC>' and 'id<MyProtocolB>')}} 

  if (obj_ab == obj_ac) foo (); // expected-warning {{comparison of distinct pointer types ('id<MyProtocolAB>' and 'id<MyProtocolAC>')}} 
  if (obj_ac == obj_ab) foo (); // expected-warning {{comparison of distinct pointer types ('id<MyProtocolAC>' and 'id<MyProtocolAB>')}} 

  return 0;
}
