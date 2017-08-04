// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -analyzer-checker=core,valist.Uninitialized,valist.CopyToSelf -analyzer-output=text -analyzer-store=region -verify %s

#include "Inputs/system-header-simulator-for-valist.h"

// This is called in call_inlined_uses_arg(),
// and the warning is generated during the analysis of call_inlined_uses_arg().
void inlined_uses_arg(va_list arg) {
  (void)va_arg(arg, int); // expected-warning{{va_arg() is called on an uninitialized va_list}}
  // expected-note@-1{{va_arg() is called on an uninitialized va_list}}
}

void call_inlined_uses_arg(int fst, ...) {
  va_list va;
  inlined_uses_arg(va); // expected-note{{Calling 'inlined_uses_arg'}}
}

void f6(va_list *fst, ...) {
  va_start(*fst, fst);
  // FIXME: There should be no warning for this.
  (void)va_arg(*fst, int); // expected-warning{{va_arg() is called on an uninitialized va_list}}
  // expected-note@-1{{va_arg() is called on an uninitialized va_list}}
  va_end(*fst);
} 

void call_vprintf_bad(int isstring, ...) {
  va_list va;
  vprintf(isstring ? "%s" : "%d", va); // expected-warning{{Function 'vprintf' is called with an uninitialized va_list argument}}
  // expected-note@-1{{Function 'vprintf' is called with an uninitialized va_list argument}}
  // expected-note@-2{{Assuming 'isstring' is 0}}
  // expected-note@-3{{'?' condition is false}}
}

void call_vsprintf_bad(char *buffer, ...) {
  va_list va;
  va_start(va, buffer); // expected-note{{Initialized va_list}}
  va_end(va); // expected-note{{Ended va_list}}
  vsprintf(buffer, "%s %d %d %lf %03d", va); // expected-warning{{Function 'vsprintf' is called with an uninitialized va_list argument}}
  // expected-note@-1{{Function 'vsprintf' is called with an uninitialized va_list argument}}
}

