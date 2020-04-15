// RUN: %clang_analyze_cc1 -verify %s -fblocks \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-output=text

int* stack_addr_escape_base() {
  int x = 0;
  // FIXME: This shouldn't be tied to a modeling checker.
  return &x; // expected-warning{{Address of stack memory associated with local variable 'x' returned to caller [core.StackAddrEscapeBase]}}
  // expected-note-re@-1{{{{^Address of stack memory associated with local variable 'x' returned to caller$}}}}
  // Just a regular compiler warning.
  // expected-warning@-3{{address of stack memory associated with local variable 'x' returned}}
}

