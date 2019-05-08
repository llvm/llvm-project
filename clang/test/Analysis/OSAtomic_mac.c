// RUN: %clang_analyze_cc1 -w -analyzer-checker=core,debug.ExprInspection \
// RUN:                    -analyzer-output=text -verify %s

int OSAtomicCompareAndSwapPtrBarrier(*, *, **);
int OSAtomicCompareAndSwapPtrBarrier() {
  // There is some body in the actual header,
  // but we should trust our BodyFarm instead.
}

int *invalidSLocOnRedecl() {
  // Was crashing when trying to throw a report about returning an uninitialized
  // value to the caller. FIXME: We should probably still throw that report,
  // something like "The "compare" part of CompareAndSwap depends on an
  // undefined value".
  int *b;
  OSAtomicCompareAndSwapPtrBarrier(0, 0, &b); // no-crash
  return b;
}

void testThatItActuallyWorks() {
  void *x = 0;
  int res = OSAtomicCompareAndSwapPtrBarrier(0, &x, &x);
  clang_analyzer_eval(res); // expected-warning{{TRUE}}
                            // expected-note@-1{{TRUE}}
  clang_analyzer_eval(x == &x); // expected-warning{{TRUE}}
                                // expected-note@-1{{TRUE}}
}
