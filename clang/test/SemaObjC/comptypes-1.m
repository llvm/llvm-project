// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

#define nil (void *)0;
#define Nil (void *)0;

extern void foo();

@protocol MyProtocol
- (void) foo;
@end

@interface MyClass
@end

@interface MyOtherClass <MyProtocol>
- (void) foo;
@end

int main()
{
  id obj = nil;
  id<MyProtocol> obj_p = nil;
  MyClass *obj_c = nil;
  MyOtherClass *obj_cp = nil;
  Class obj_C = Nil;

  /* Assigning to an 'id' variable should never
     generate a warning.  */
  obj = obj_p;  /* Ok  */
  obj = obj_c;  /* Ok  */
  obj = obj_cp; /* Ok  */
  obj = obj_C;  /* Ok  */
  
  /* Assigning to a 'MyClass *' variable should always generate a
     warning, unless done from an 'id'.  */
  obj_c = obj;    /* Ok */
  obj_c = obj_cp; // // expected-warning {{incompatible pointer types assigning to 'MyClass *' from 'MyOtherClass *'}}
  obj_c = obj_C;  // expected-warning {{incompatible pointer types assigning to 'MyClass *' from 'Class'}}

  /* Assigning to an 'id<MyProtocol>' variable should generate a
     warning if done from a 'MyClass *' (which doesn't implement
     MyProtocol), but not from an 'id' or from a 'MyOtherClass *'
     (which implements MyProtocol).  */
  obj_p = obj;    /* Ok */
  obj_p = obj_c;  // expected-warning {{assigning to 'id<MyProtocol>' from incompatible type 'MyClass *'}}
  obj_p = obj_cp; /* Ok  */
  obj_p = obj_C;  // expected-warning {{incompatible pointer types assigning to 'id<MyProtocol>' from 'Class'}}

  /* Assigning to a 'MyOtherClass *' variable should always generate
     a warning, unless done from an 'id' or an 'id<MyProtocol>' (since
     MyOtherClass implements MyProtocol).  */
  obj_cp = obj;    /* Ok */
  obj_cp = obj_c;  // expected-warning {{incompatible pointer types assigning to 'MyOtherClass *' from 'MyClass *'}}
  obj_cp = obj_p;  /* Ok */
  obj_cp = obj_C;  // expected-warning {{incompatible pointer types assigning to 'MyOtherClass *' from 'Class'}}

  /* Any comparison involving an 'id' must be without warnings.  */
  if (obj == obj_p) foo() ;  /* Ok  */ /*Bogus warning here in 2.95.4*/
  if (obj_p == obj) foo() ;  /* Ok  */
  if (obj == obj_c) foo() ;  /* Ok  */
  if (obj_c == obj) foo() ;  /* Ok  */
  if (obj == obj_cp) foo() ; /* Ok  */
  if (obj_cp == obj) foo() ; /* Ok  */
  if (obj == obj_C) foo() ;  /* Ok  */
  if (obj_C == obj) foo() ;  /* Ok  */

  /* Any comparison between 'MyClass *' and anything which is not an 'id'
     must generate a warning.  */
  if (obj_p == obj_c) foo() ; // expected-warning {{comparison of distinct pointer types ('id<MyProtocol>' and 'MyClass *')}}

  if (obj_c == obj_cp) foo() ; // expected-warning {{comparison of distinct pointer types ('MyClass *' and 'MyOtherClass *')}} 
  if (obj_cp == obj_c) foo() ; // expected-warning {{comparison of distinct pointer types ('MyOtherClass *' and 'MyClass *')}}

  if (obj_c == obj_C) foo() ;
  if (obj_C == obj_c) foo() ;

  /* Any comparison between 'MyOtherClass *' (which implements
     MyProtocol) and an 'id' implementing MyProtocol are Ok.  */
  if (obj_cp == obj_p) foo() ; /* Ok */
  if (obj_p == obj_cp) foo() ; /* Ok */


  if (obj_p == obj_C) foo() ;  
  if (obj_C == obj_p) foo() ;
  if (obj_cp == obj_C) foo() ;  
  if (obj_C == obj_cp) foo() ;

  return 0;
}
