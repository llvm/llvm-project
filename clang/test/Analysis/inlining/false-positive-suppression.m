// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config suppress-null-return-paths=false -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -verify -DSUPPRESSED=1 %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-config avoid-suppressing-null-argument-paths=true -DSUPPRESSED=1 -DNULL_ARGS=1 -verify %s

#ifdef SUPPRESSED
// expected-no-diagnostics
#endif

@interface PointerWrapper
- (int *)getPtr;
- (id)getObject;
@end

id getNil() {
  return 0;
}

void testNilReceiverHelperA(int *x) {
  *x = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testNilReceiverHelperB(int *x) {
  *x = 1;
#ifndef SUPPRESSED
  // expected-warning@-2 {{Dereference of null pointer}}
#endif
}

void testNilReceiver(int coin) {
  id x = getNil();
  if (coin)
    testNilReceiverHelperA([x getPtr]);
  else
    testNilReceiverHelperB([[x getObject] getPtr]);
}
