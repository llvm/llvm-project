// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -analyzer-max-loop 4 -verify %s
#include "Inputs/system-header-simulator.h"

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

static int another_function(int *y) {
  if (*y > 0)
    return *y;
  return 0;
}

static void function_which_doesnt_give_up(int **x) {
   *x = 0;
}

static void function_which_gives_up(int *x) {
  for (int i = 0; i < 5; ++i)
    (*x)++;
}

static void function_which_gives_up_nested(int *x) {
  function_which_gives_up(x);
  for (int i = 0; i < 5; ++i)
    (*x)++;
}

static void function_which_doesnt_give_up_nested(int *x, int *y) {
  *y = another_function(x);
  function_which_gives_up(x);
}

void coverage1(int *x) {
  function_which_gives_up(x);
  char *m = (char*)malloc(12);
} // expected-warning {{Potential leak of memory pointed to by 'm'}}

void coverage2(int *x) {
  if (x) {
    function_which_gives_up(x);
    char *m = (char*)malloc(12);
  }
} // expected-warning {{Potential leak of memory pointed to by 'm'}}

void coverage3(int *x) {
  x++;
  function_which_gives_up(x);
  char *m = (char*)malloc(12);
} // expected-warning {{Potential leak of memory pointed to by 'm'}}

void coverage4(int *x) {
  *x += another_function(x);
  function_which_gives_up(x);
  char *m = (char*)malloc(12);
} // expected-warning {{Potential leak of memory pointed to by 'm'}}

void coverage5(int *x) {
  for (int i = 0; i<7; ++i)
    function_which_gives_up(x);
  // The root function gives up here.
  char *m = (char*)malloc(12); // no-warning
}

void coverage6(int *x) {
  for (int i = 0; i<3; ++i) {
    function_which_gives_up(x);
  }
  char *m = (char*)malloc(12);
} // expected-warning {{Potential leak of memory pointed to by 'm'}}

int coverage7_inline(int *i) {
  function_which_doesnt_give_up(&i);
  return *i; // expected-warning {{Dereference}}
}

void coverage8(int *x) {
  int y;
  function_which_doesnt_give_up_nested(x, &y);
  y = (*x)/y;  // expected-warning {{Division by zero}}
  char *m = (char*)malloc(12);
} // expected-warning {{Potential leak of memory pointed to by 'm'}}

void function_which_gives_up_settonull(int **x) {
  *x = 0;
  int y = 0;
  for (int i = 0; i < 5; ++i)
    y++;
}

void coverage9(int *x) {
  int y = 5;
  function_which_gives_up_settonull(&x);
  y = (*x);  // no warning
}

static void empty_function(void){
}
int use_empty_function(int x) {
    x = 0;
    empty_function();
    return 5/x; //expected-warning {{Division by zero}}
}
