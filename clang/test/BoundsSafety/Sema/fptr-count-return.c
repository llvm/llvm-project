
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -ast-dump -verify %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -verify %s | FileCheck %s
#include <ptrcheck.h>

// CHECK: VarDecl {{.*}} fptr 'int *__single*__single __counted_by(len)(*__single)(int)'
int **__counted_by(len) (*fptr)(int len);

// expected-error@+1{{cannot apply '__counted_by' attribute to 'void *' because 'void' has unknown size; did you mean to use '__sized_by' instead?}}
void *__counted_by(len) (*fptr1)(int len);

// expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
int *__counted_by(len)* (*fptr2)(int len);

// CHECK: VarDecl {{.*}} fptr3 'int *__single __sized_by(len1)(*__single)(int)'
// CHECK: VarDecl {{.*}} fptr4 'int *__single __sized_by(len2)(*__single)(void *__single __sized_by(len2), int)'
int *__sized_by(len1)(*fptr3)(int len1), *__sized_by(len2)(*fptr4)(void *__sized_by(len2), int len2);

// CHECK: VarDecl {{.*}} fptr5 'void *__single __sized_by(len1)(*__single)(int)'
void *__sized_by(len1) (*fptr5)(int len1),

// expected-error@+2{{use of undeclared identifier 'len1'; did you mean 'len2'?}}
// expected-note@+1{{'len2' declared here}}
     *__sized_by(len1) (*fptr6)(void *__sized_by(len2), unsigned len2);

int glen;
// CHECK: VarDecl {{.*}} fptr7 'void *__single __sized_by(glen)(*__single)(int)'
void *__sized_by(glen) (*fptr7)(int glen); // ok. glen shadowed.

// expected-error@+1{{argument of '__sized_by' attribute cannot refer to declaration from a different scope}}
void *__sized_by(glen) (*fptr8)();


// CHECK: RecordDecl {{.*}} struct T1 definition
// CHECK: |-FieldDecl {{.*}} len 'int'
// CHECK: `-FieldDecl {{.*}} fptr 'void *__single __sized_by(len)(*__single)(int)'
struct T1 {
  int len;
  void *__sized_by(len) (*fptr)(int len);
};

struct T2 {
  int field_len;
  // expected-error@+1{{use of undeclared identifier 'field_len'}}
  int *__sized_by(field_len) (*fptr)(int len);
};

struct T3 {
  int field_len;
  // expected-error@+1{{use of undeclared identifier 'field_len'}}
  void *__sized_by(field_len) (*fptr)();
};

// CHECK: RecordDecl {{.*}} struct T4 definition
// CHECK:  |-FieldDecl {{.*}} fptr1 'void *__single __sized_by(len1)(*__single)(unsigned int)'
// CHECK:  |-FieldDecl {{.*}} fptr2 'void *__single __sized_by(len2)(*__single)(unsigned int)'
// CHECK:  |-FieldDecl {{.*}} fptr3 'void *__single __sized_by(len3)(*__single)(unsigned int)'
struct T4 {
  void *__sized_by(len1) (*fptr1)(unsigned len1),
       *__sized_by(len2) (*fptr2)(unsigned len2);

  void *__sized_by(len3) (*fptr3)(unsigned len3),
  // expected-error@+2{{use of undeclared identifier 'len3'; did you mean 'len4'?}}
  // expected-note@+1{{'len4' declared here}}
       *__sized_by(len3) (*fptr4)(unsigned len4);
};
