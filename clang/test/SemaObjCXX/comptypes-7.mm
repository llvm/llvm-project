// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

#define nil nullptr

extern void foo();

@protocol MyProtocol
- (void) method;
@end

@interface MyClass
@end

int main()
{
  id obj = nil;
  id <MyProtocol> obj_p = nil;
  MyClass *obj_c = nil;
  Class obj_C = nil;

  int i = 0;
  int *j = nil;

  /* These should all generate errors.  */

  obj = i; // expected-error {{assigning to 'id' from incompatible type 'int'}}
  obj = j; // expected-error {{assigning to 'id' from incompatible type 'int *'}}

  obj_p = i; // expected-error {{assigning to 'id<MyProtocol>' from incompatible type 'int'}}
  obj_p = j; // expected-error {{assigning to 'id<MyProtocol>' from incompatible type 'int *'}}

  obj_c = i; // expected-error {{assigning to 'MyClass *' from incompatible type 'int'}}
  obj_c = j; // expected-error {{assigning to 'MyClass *' from incompatible type 'int *'}}

  obj_C = i; // expected-error {{assigning to 'Class' from incompatible type 'int'}}
  obj_C = j; // expected-error {{assigning to 'Class' from incompatible type 'int *'}}

  i = obj;   // expected-error {{assigning to 'int' from incompatible type 'id'}}
  i = obj_p; // expected-error {{assigning to 'int' from incompatible type 'id<MyProtocol>'}}
  i = obj_c; // expected-error {{assigning to 'int' from incompatible type 'MyClass *'}}
  i = obj_C; // expected-error {{assigning to 'int' from incompatible type 'Class'}}

  j = obj;   // expected-error {{assigning to 'int *' from incompatible type 'id'}}
  j = obj_p; // expected-error {{assigning to 'int *' from incompatible type 'id<MyProtocol>'}}
  j = obj_c; // expected-error {{assigning to 'int *' from incompatible type 'MyClass *'}}
  j = obj_C; // expected-error {{assigning to 'int *' from incompatible type 'Class'}}

  if (obj == i) foo() ; // expected-error {{comparison between pointer and integer ('id' and 'int')}}
  if (i == obj) foo() ; // expected-error {{comparison between pointer and integer ('int' and 'id')}}
  if (obj == j) foo() ; // expected-error {{comparison of distinct pointer types ('id' and 'int *')}}
  if (j == obj) foo() ; // expected-error {{comparison of distinct pointer types ('int *' and 'id')}}

  if (obj_c == i) foo() ; // expected-error {{comparison between pointer and integer ('MyClass *' and 'int')}}
  if (i == obj_c) foo() ; // expected-error {{comparison between pointer and integer ('int' and 'MyClass *')}}
  if (obj_c == j) foo() ; // expected-error {{comparison of distinct pointer types ('MyClass *' and 'int *')}}
  if (j == obj_c) foo() ; // expected-error {{comparison of distinct pointer types ('int *' and 'MyClass *')}}

  if (obj_p == i) foo() ; // expected-error {{comparison between pointer and integer ('id<MyProtocol>' and 'int')}}
  if (i == obj_p) foo() ; // expected-error {{comparison between pointer and integer ('int' and 'id<MyProtocol>')}}
  if (obj_p == j) foo() ; // expected-error {{comparison of distinct pointer types ('id<MyProtocol>' and 'int *')}}
  if (j == obj_p) foo() ; // expected-error {{comparison of distinct pointer types ('int *' and 'id<MyProtocol>')}}

  if (obj_C == i) foo() ; // expected-error {{comparison between pointer and integer ('Class' and 'int')}}
  if (i == obj_C) foo() ; // expected-error {{comparison between pointer and integer ('int' and 'Class')}}
  if (obj_C == j) foo() ; // expected-error {{comparison of distinct pointer types ('Class' and 'int *')}}
  if (j == obj_C) foo() ; // expected-error {{comparison of distinct pointer types ('int *' and 'Class')}}

  Class bar1 = nil;
  Class <MyProtocol> bar = nil;
  bar = bar1;
  bar1 = bar;

  return 0;
}
