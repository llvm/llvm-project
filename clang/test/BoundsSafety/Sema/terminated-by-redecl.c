
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// expected-note@+1{{previous declaration is here}}
char *test();
// expected-error@+1{{conflicting '__terminated_by' attribute with the previous function declaration}}
char *__null_terminated test();

// expected-note@+1{{previous declaration is here}}
const char *test2();
// expected-error@+1{{conflicting '__terminated_by' attribute with the previous function declaration}}
const char *__terminated_by(-1) test2();

const char *test3();
const char *__null_terminated test3();

// expected-note@+1{{previous declaration is here}}
void test4(int *__null_terminated arg);
// expected-error@+1{{conflicting '__terminated_by' attribute with the previous function declaration}}
void test4(int *arg);

// expected-note@+1{{previous declaration is here}}
void test5(int *__null_terminated arg);
// expected-error@+1{{conflicting '__terminated_by' attribute with the previous function declaration}}
void test5(int *__terminated_by(-1) arg);

// expected-note@+1{{previous declaration is here}}
void test6(int *__counted_by(len) buf, int len);
// expected-error@+1{{conflicting '__terminated_by' attribute with the previous function declaration}}
void test6(int *__null_terminated buf, int len);

// expected-note@+1{{previous declaration is here}}
void test7(int *__null_terminated buf, int len);
// expected-error@+1{{conflicting '__sized_by' attribute with the previous function declaration}}
void test7(int *__sized_by(len) buf, int len);

// expected-note@+1{{previous declaration is here}}
void test8(int *__null_terminated buf, int len);
// expected-error@+1{{conflicting '__ended_by' attribute with the previous function declaration}}
void test8(int *__ended_by(end) buf, int *end);

// expected-note@+1{{previous declaration is here}}
void test9(int *__ended_by(end) buf, int *end);
// expected-error@+1{{conflicting '__ended_by' attribute with the previous function declaration}}
void test9(int *buf, int *__null_terminated end);

#include "terminated-by-redecl.h"

// expected-error@+1{{conflicting '__terminated_by' attribute with the previous function declaration}}
void test_system_nt_argument(int *p);
// expected-error@+1{{conflicting '__terminated_by' attribute with the previous function declaration}}
int *test_system_nt_return();
