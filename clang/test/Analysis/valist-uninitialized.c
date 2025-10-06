// RUN: %clang_analyze_cc1 -triple hexagon-unknown-linux -verify %s \
// RUN:   -analyzer-checker=core,security.VAList \
// RUN:   -analyzer-disable-checker=core.CallAndMessage \
// RUN:   -analyzer-output=text
//
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -verify %s \
// RUN:   -analyzer-checker=core,security.VAList \
// RUN:   -analyzer-disable-checker=core.CallAndMessage \
// RUN:   -analyzer-output=text
//
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu %s \
// RUN:   -analyzer-checker=core,security.VAList

#include "Inputs/system-header-simulator-for-valist.h"

void f1(int fst, ...) {
  va_list va;
  (void)va_arg(va, int); // expected-warning{{va_arg() is called on an uninitialized va_list}}
  // expected-note@-1{{va_arg() is called on an uninitialized va_list}}
}

int f2(int fst, ...) {
  va_list va;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_end(va); // expected-note{{Ended va_list}}
  return va_arg(va, int); // expected-warning{{va_arg() is called on an already released va_list}}
  // expected-note@-1{{va_arg() is called on an already released va_list}}
}

void f3(int fst, ...) {
  va_list va, va2;
  va_start(va, fst);
  va_copy(va2, va);
  va_end(va);
  (void)va_arg(va2, int);
  va_end(va2);
} //no-warning

void f4(int cond, ...) {
  va_list va;
  if (cond) { // expected-note{{Assuming 'cond' is 0}}
    // expected-note@-1{{Taking false branch}}
    va_start(va, cond);
    (void)va_arg(va,int);
  }
  va_end(va); //expected-warning{{va_end() is called on an uninitialized va_list}}
  // expected-note@-1{{va_end() is called on an uninitialized va_list}}
}

void f7(int *fst, ...) {
  va_list x;
  va_list *y = &x;
  va_start(*y,fst);
  (void)va_arg(x, int);
  va_end(x);
} // no-warning

void f8(int *fst, ...) {
  va_list x;
  va_list *y = &x;
  va_start(*y,fst); // expected-note{{Initialized va_list}}
  va_end(x); // expected-note{{Ended va_list}}
  (void)va_arg(*y, int); //expected-warning{{va_arg() is called on an already released va_list}}
  // expected-note@-1{{va_arg() is called on an already released va_list}}
}

void reinitOk(int *fst, ...) {
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int);
  va_end(va);
  va_start(va, fst);
  (void)va_arg(va, int);
  va_end(va);
} // no-warning

void reinit3(int *fst, ...) {
  va_list va;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  (void)va_arg(va, int);
  va_end(va); // expected-note{{Ended va_list}}
  va_start(va, fst); // expected-note{{Initialized va_list}}
  (void)va_arg(va, int);
  va_end(va); // expected-note{{Ended va_list}}
  (void)va_arg(va, int); //expected-warning{{va_arg() is called on an already released va_list}}
  // expected-note@-1{{va_arg() is called on an already released va_list}}
}

void copyUnint(int fst, ...) {
  va_list va, va2;
  va_copy(va, va2); // expected-warning{{Uninitialized va_list is copied}}
  // expected-note@-1{{Uninitialized va_list is copied}}
}

void g1(int fst, ...) {
  va_list va;
  va_end(va); // expected-warning{{va_end() is called on an uninitialized va_list}}
  // expected-note@-1{{va_end() is called on an uninitialized va_list}}
}

void g2(int fst, ...) {
  va_list va;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_end(va); // expected-note{{Ended va_list}}
  va_end(va); // expected-warning{{va_end() is called on an already released va_list}}
  // expected-note@-1{{va_end() is called on an already released va_list}}
}

void is_sink(int fst, ...) {
  va_list va;
  va_end(va); // expected-warning{{va_end() is called on an uninitialized va_list}}
  // expected-note@-1{{va_end() is called on an uninitialized va_list}}
  *((volatile int *)0) = 1;
}

// NOTE: this is invalid, as the man page of va_end requires that "Each invocation of va_start()
// must be matched by a corresponding  invocation of va_end() in the same function."
void ends_arg(va_list arg) {
  va_end(arg);
} //no-warning

void uses_arg(va_list arg) {
  (void)va_arg(arg, int);
} //no-warning

void call_vprintf_ok(int isstring, ...) {
  va_list va;
  va_start(va, isstring);
  vprintf(isstring ? "%s" : "%d", va);
  va_end(va);
} //no-warning

void call_some_other_func(int n, ...) {
  va_list va;
  some_library_function(n, va);
} //no-warning

void inlined_uses_arg_good(va_list arg) {
  (void)va_arg(arg, int);
}

void call_inlined_uses_arg_good(int fst, ...) {
  va_list va;
  va_start(va, fst);
  inlined_uses_arg_good(va);
  va_end(va);
}

void va_copy_test(va_list arg) {
  va_list dst;
  va_copy(dst, arg);
  va_end(dst);
}

void all_state_changes(va_list unknown, int fst, ...) {
  va_list va, va2;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_copy(va, unknown); // expected-note{{Copied unknown contents into the va_list}}
  va_end(va); // expected-note{{Ended va_list}}
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_copy(va, va2); // expected-note{{Copied uninitialized contents into the va_list}}
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_end(va); // expected-note{{Ended va_list}}
  va_end(va); // expected-warning{{va_end() is called on an already released va_list}}
  // expected-note@-1{{va_end() is called on an already released va_list}}
}
