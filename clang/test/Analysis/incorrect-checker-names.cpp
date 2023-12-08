// RUN: %clang_analyze_cc1 -verify %s -fblocks \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-output=text

int* stack_addr_escape_base() {
  int x = 0;
  // FIXME: This shouldn't be tied to a modeling checker.
  return &x; // expected-warning{{Address of stack memory associated with local variable 'x' returned to caller [core.StackAddressEscape]}}
  // expected-note-re@-1{{{{^Address of stack memory associated with local variable 'x' returned to caller$}}}}
  // Just a regular compiler warning.
  // expected-warning@-3{{address of stack memory associated with local variable 'x' returned}}
}

char const *p;

void f0() {
  char const str[] = "This will change";
  p = str;
} // expected-warning{{Address of stack memory associated with local variable 'str' is still referred to by the global variable 'p' upon returning to the caller.  This will be a dangling reference [core.StackAddressEscape]}}
// expected-note@-1{{Address of stack memory associated with local variable 'str' is still referred to by the global variable 'p' upon returning to the caller.  This will be a dangling reference}}
