// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol Foo;

Class T;
id<Foo> S;
id R;
void foo(void) {
  // Test assignment compatibility of Class and id.  No warning should be
  // produced.
  // rdar://6770142 - Class and id<foo> are compatible.
  S = T; // expected-warning {{incompatible pointer types assigning to 'id<Foo>' from 'Class'}}
  T = S; // expected-warning {{incompatible pointer types assigning to 'Class' from 'id<Foo>'}}
  R = T; T = R;
  R = S; S = R;
}

// Test attempt to redefine 'id' in an incompatible fashion.
// rdar://11356439
typedef int id;  // expected-error {{typedef redefinition with different types ('int' vs 'id')}}
id b;

typedef double id;  // expected-error {{typedef redefinition with different types ('double' vs 'id')}}

typedef char *id; // expected-error {{typedef redefinition with different types ('char *' vs 'id')}}

typedef union U{ int iu; } *id; // expected-error {{typedef redefinition with different types ('union U *' vs 'id')}}

void test11356439(id o) {
  o->x; // expected-error {{member reference base type 'id' is not a structure or union}}
}
