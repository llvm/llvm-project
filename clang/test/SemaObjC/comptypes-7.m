// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

#define nil (void *)0;

extern void foo(void);

@protocol MyProtocol
- (void) method;
@end

@interface MyClass
@end

int main(void)
{
  id obj = nil;
  id <MyProtocol> obj_p = nil;
  MyClass *obj_c = nil;
  Class obj_C = nil;
  
  int i = 0;
  int *j = nil;

  /* The incompatible pointer types should all generate warnings, while the
     incompatible integer to/from pointer conversions default to an error. */
  
  obj = i; // expected-error {{incompatible integer to pointer conversion assigning to 'id' from 'int'}}
  obj = j; // expected-warning {{incompatible pointer types assigning to 'id' from 'int *'}}

  obj_p = i; // expected-error {{incompatible integer to pointer conversion assigning to 'id<MyProtocol>' from 'int'}}
  obj_p = j; // expected-warning {{incompatible pointer types assigning to 'id<MyProtocol>' from 'int *'}}
  
  obj_c = i; // expected-error {{incompatible integer to pointer conversion assigning to 'MyClass *' from 'int'}}
  obj_c = j; // expected-warning {{incompatible pointer types assigning to 'MyClass *' from 'int *'}}

  obj_C = i; // expected-error {{incompatible integer to pointer conversion assigning to 'Class' from 'int'}}
  obj_C = j; // expected-warning {{incompatible pointer types assigning to 'Class' from 'int *'}}
  
  i = obj;   // expected-error {{incompatible pointer to integer conversion assigning to 'int' from 'id'}}
  i = obj_p; // expected-error {{incompatible pointer to integer conversion assigning to 'int' from 'id<MyProtocol>'}}
  i = obj_c; // expected-error {{incompatible pointer to integer conversion assigning to 'int' from 'MyClass *'}}
  i = obj_C; // expected-error {{incompatible pointer to integer conversion assigning to 'int' from 'Class'}}
  
  j = obj;   // expected-warning {{incompatible pointer types assigning to 'int *' from 'id'}}
  j = obj_p; // expected-warning {{incompatible pointer types assigning to 'int *' from 'id<MyProtocol>'}}
  j = obj_c; // expected-warning {{incompatible pointer types assigning to 'int *' from 'MyClass *'}}
  j = obj_C; // expected-warning {{incompatible pointer types assigning to 'int *' from 'Class'}}
  
  if (obj == i) foo() ; // expected-warning {{comparison between pointer and integer ('id' and 'int')}}
  if (i == obj) foo() ; // expected-warning {{comparison between pointer and integer ('int' and 'id')}}
  if (obj == j) foo() ; // expected-warning {{comparison of distinct pointer types ('id' and 'int *')}}
  if (j == obj) foo() ; // expected-warning {{comparison of distinct pointer types ('int *' and 'id')}}

  if (obj_c == i) foo() ; // expected-warning {{comparison between pointer and integer ('MyClass *' and 'int')}}
  if (i == obj_c) foo() ; // expected-warning {{comparison between pointer and integer ('int' and 'MyClass *')}}
  if (obj_c == j) foo() ; // expected-warning {{comparison of distinct pointer types ('MyClass *' and 'int *')}}
  if (j == obj_c) foo() ; // expected-warning {{comparison of distinct pointer types ('int *' and 'MyClass *')}}

  if (obj_p == i) foo() ; // expected-warning {{comparison between pointer and integer ('id<MyProtocol>' and 'int')}}
  if (i == obj_p) foo() ; // expected-warning {{comparison between pointer and integer ('int' and 'id<MyProtocol>')}}
  if (obj_p == j) foo() ; // expected-warning {{comparison of distinct pointer types ('id<MyProtocol>' and 'int *')}}
  if (j == obj_p) foo() ; // expected-warning {{comparison of distinct pointer types ('int *' and 'id<MyProtocol>')}}

  if (obj_C == i) foo() ; // expected-warning {{comparison between pointer and integer ('Class' and 'int')}}
  if (i == obj_C) foo() ; // expected-warning {{comparison between pointer and integer ('int' and 'Class')}}
  if (obj_C == j) foo() ; // expected-warning {{comparison of distinct pointer types ('Class' and 'int *')}}
  if (j == obj_C) foo() ; // expected-warning {{comparison of distinct pointer types ('int *' and 'Class')}}

  Class bar1 = nil;
  Class <MyProtocol> bar = nil;
  bar = bar1;
  bar1 = bar;

  return 0;
}
