// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify -std=c99 -Dbool=_Bool -Wno-bool-conversion %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify -x c++ -Wno-bool-conversion %s

typedef __INTPTR_TYPE__ intptr_t;
char const *p;

void f0() {
  char const str[] = "This will change";
  p = str;
}  // expected-warning{{Address of stack memory associated with local variable 'str' is still referred to by the global variable 'p' upon returning to the caller.  This will be a dangling reference}}

void f1() {
  char const str[] = "This will change";
  p = str; 
  p = 0; // no-warning
}

void f2() {
  p = (const char *) __builtin_alloca(12);
} // expected-warning{{Address of stack memory allocated by call to alloca() on line 19 is still referred to by the global variable 'p' upon returning to the caller.  This will be a dangling reference}}

// PR 7383 - previously the stack address checker would crash on this example
//  because it would attempt to do a direct load from 'pr7383_list'. 
static int pr7383(__const char *__)
{
  return 0;
}
extern __const char *__const pr7383_list[];

// Test that we catch multiple returns via globals when analyzing a function.
void test_multi_return() {
  static int *a, *b;
  int x;
  a = &x;
  b = &x;
} // expected-warning{{Address of stack memory associated with local variable 'x' is still referred to by the static variable 'a' upon returning}} expected-warning{{Address of stack memory associated with local variable 'x' is still referred to by the static variable 'b' upon returning}}

intptr_t returnAsNonLoc() {
  int x;
  return (intptr_t)&x; // expected-warning{{Address of stack memory associated with local variable 'x' returned to caller}}
}

bool returnAsBool() {
  int x;
  return &x; // no-warning
}

void assignAsNonLoc() {
  extern intptr_t ip;
  int x;
  ip = (intptr_t)&x;
} // expected-warning{{Address of stack memory associated with local variable 'x' is still referred to by the global variable 'ip' upon returning}}

void assignAsBool() {
  extern bool b;
  int x;
  b = &x;
} // no-warning
