// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s -std=c11

int test1(int *a) {
  return a == '\0'; // expected-warning {{comparing a pointer to a null character constant; did you mean to compare to (void *)0?}}
}

int test2(int *a) {
  return '\0' == a; // expected-warning {{comparing a pointer to a null character constant; did you mean to compare to (void *)0?}}
}

int test3(int *a) {
  return a == L'\0'; // expected-warning {{comparing a pointer to a null character constant; did you mean to compare to (void *)0?}}
}

int test4(int *a) {
  return a == u'\0'; // expected-warning {{comparing a pointer to a null character constant; did you mean to compare to (void *)0?}}
}

int test5(int *a) {
  return a == U'\0'; // expected-warning {{comparing a pointer to a null character constant; did you mean to compare to (void *)0?}}
}

int test6(int *a) {
  return a == (char)0; // expected-warning {{comparing a pointer to a null character constant; did you mean to compare to (void *)0?}}
}

typedef char my_char;
int test7(int *a) {
  return a == (my_char)0;
  // expected-warning@-1 {{comparing a pointer to a null character constant; did you mean to compare to (void *)0?}}
}

int test8(int *a) {
  return a != '\0'; // expected-warning {{comparing a pointer to a null character constant; did you mean to compare to (void *)0?}}
}

#define NULL (void *)0
int test9(int *a) {
  return a == '\0'; // expected-warning {{comparing a pointer to a null character constant; did you mean to compare to NULL?}}
}

#define MYCHAR char
int test10(int *a) {
  return a == (MYCHAR)0; // expected-warning {{comparing a pointer to a null character constant; did you mean to compare to NULL?}}
}

int test11(int *a) {
  return a > '\0';
}
