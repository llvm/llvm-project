// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -analyze -analyzer-checker=core,alpha.valist.Uninitialized,alpha.valist.CopyToSelf -analyzer-output=text -analyzer-store=region -verify %s

#include "Inputs/system-header-simulator-for-valist.h"

void f1(int fst, ...) {
  va_list va;
  (void)va_arg(va, int); //expected-warning{{va_arg() is called on an uninitialized va_list}} expected-note{{va_arg() is called on an uninitialized va_list}}
}

int f2(int fst, ...) {
  va_list va;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_end(va); // expected-note{{Ended va_list}}
  return va_arg(va, int); //expected-warning{{va_arg() is called on an uninitialized va_list}} expected-note{{va_arg() is called on an uninitialized va_list}}
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
  if (cond) { // expected-note{{Assuming 'cond' is 0}} expected-note{{Taking false branch}}
    va_start(va, cond);
    (void)va_arg(va,int);
  }
  va_end(va); //expected-warning{{va_end() is called on an uninitialized va_list}} expected-note{{va_end() is called on an uninitialized va_list}}
}

void f5(va_list fst, ...) {
  va_start(fst, fst);
  (void)va_arg(fst, int);
  va_end(fst);
} // no-warning

//FIXME: this should not cause a warning
void f6(va_list *fst, ...) {
  va_start(*fst, fst);
  (void)va_arg(*fst, int); //expected-warning{{va_arg() is called on an uninitialized va_list}} expected-note{{va_arg() is called on an uninitialized va_list}}
  va_end(*fst);
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
  (void)va_arg(*y, int); //expected-warning{{va_arg() is called on an uninitialized va_list}} expected-note{{va_arg() is called on an uninitialized va_list}}
} // no-warning

// This only contains problems which are handled by varargs.Unterminated.
void reinit(int *fst, ...) {
  va_list va;
  va_start(va, fst);
  va_start(va, fst);
  (void)va_arg(va, int);
} // no-warning

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
  (void)va_arg(va, int); //expected-warning{{va_arg() is called on an uninitialized va_list}} expected-note{{va_arg() is called on an uninitialized va_list}}
}

void copyself(int fst, ...) {
  va_list va;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_copy(va, va); // expected-warning{{va_list 'va' is copied onto itself}} expected-note{{va_list 'va' is copied onto itself}}
  va_end(va);
} // no-warning

void copyselfUninit(int fst, ...) {
  va_list va;
  va_copy(va, va); // expected-warning{{va_list 'va' is copied onto itself}} expected-note{{va_list 'va' is copied onto itself}}
} // no-warning

void copyOverwrite(int fst, ...) {
  va_list va, va2;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_copy(va, va2); // expected-warning{{Initialized va_list 'va' is overwritten by an uninitialized one}} expected-note{{Initialized va_list 'va' is overwritten by an uninitialized one}}
} // no-warning

void copyUnint(int fst, ...) {
  va_list va, va2;
  va_copy(va, va2); // expected-warning{{Uninitialized va_list is copied}} expected-note{{Uninitialized va_list is copied}}
}

void g1(int fst, ...) {
  va_list va;
  va_end(va); // expected-warning{{va_end() is called on an uninitialized va_list}} expected-note{{va_end() is called on an uninitialized va_list}}
}

void g2(int fst, ...) {
  va_list va;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_end(va); // expected-note{{Ended va_list}}
  va_end(va); // expected-warning{{va_end() is called on an uninitialized va_list}} expected-note{{va_end() is called on an uninitialized va_list}}
}

void is_sink(int fst, ...) {
  va_list va;
  va_end(va); // expected-warning{{va_end() is called on an uninitialized va_list}} expected-note{{va_end() is called on an uninitialized va_list}}
  *((volatile int *)0) = 1; //no-warning
}

// NOTE: this is invalid, as the man page of va_end requires that "Each invocation of va_start()
// must be matched by a corresponding  invocation of va_end() in the same function."
void ends_arg(va_list arg) {
  va_end(arg);
} //no-warning

void uses_arg(va_list arg) {
  (void)va_arg(arg, int);
} //no-warning

// This is the same function as the previous one, but it is called in call_uses_arg2(),
// and the warning is generated during the analysis of call_uses_arg2().
void inlined_uses_arg(va_list arg) {
  (void)va_arg(arg, int); //expected-warning{{va_arg() is called on an uninitialized va_list}} expected-note{{va_arg() is called on an uninitialized va_list}}
}

void call_inlined_uses_arg(int fst, ...) {
  va_list va;
  inlined_uses_arg(va); // expected-note{{Calling 'inlined_uses_arg'}}
}

void call_vprintf_ok(int isstring, ...) {
  va_list va;
  va_start(va, isstring);
  vprintf(isstring ? "%s" : "%d", va);
  va_end(va);
} //no-warning

void call_vprintf_bad(int isstring, ...) {
  va_list va;
  vprintf(isstring ? "%s" : "%d", va); //expected-warning{{Function 'vprintf' is called with an uninitialized va_list argument}} expected-note{{Function 'vprintf' is called with an uninitialized va_list argument}} expected-note{{Assuming 'isstring' is 0}} expected-note{{'?' condition is false}}
}

void call_vsprintf_bad(char *buffer, ...) {
  va_list va;
  va_start(va, buffer); // expected-note{{Initialized va_list}}
  va_end(va); // expected-note{{Ended va_list}}
  vsprintf(buffer, "%s %d %d %lf %03d", va); //expected-warning{{Function 'vsprintf' is called with an uninitialized va_list argument}} expected-note{{Function 'vsprintf' is called with an uninitialized va_list argument}}
}

void call_some_other_func(int n, ...) {
  va_list va;
  some_library_function(n, va);
} //no-warning

