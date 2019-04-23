// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,debug.ExprInspection \
// RUN:                    -analyzer-output=text -verify %s

int OSAtomicCompareAndSwapPtrBarrier(*, *, **);
int OSAtomicCompareAndSwapPtrBarrier() {
  // There is some body in the actual header,
  // but we should trust our BodyFarm instead.
}

int *invalidSLocOnRedecl() {
  int *b; // expected-note{{'b' declared without an initial value}}

  OSAtomicCompareAndSwapPtrBarrier(0, 0, &b); // no-crash
  // FIXME: We don't really need these notes.
  // expected-note@-2{{Calling 'OSAtomicCompareAndSwapPtrBarrier'}}
  // expected-note@-3{{Returning from 'OSAtomicCompareAndSwapPtrBarrier'}}

  return b; // expected-warning{{Undefined or garbage value returned to caller}}
            // expected-note@-1{{Undefined or garbage value returned to caller}}
}
