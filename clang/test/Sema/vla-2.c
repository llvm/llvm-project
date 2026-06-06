// RUN: %clang_cc1 %s -verify -fsyntax-only -pedantic
// Check that we don't crash trying to emit warnings in a potentially-evaluated
// sizeof or typeof.  (This test needs to be in a separate file because we use
// a different codepath when we have already emitted an error.)

int PotentiallyEvaluatedSizeofWarn(int n) {
  return (int)sizeof *(0 << 32,(int(*)[n])0); /* expected-warning {{shift count >= width of type}}
                                                 expected-warning {{left operand of comma operator has no effect}} */
}

void PotentiallyEvaluatedTypeofWarn(int n) {
  __typeof(*(0 << 32,(int(*)[n])0)) x; /* expected-warning {{shift count >= width of type}}
                                          expected-warning {{left operand of comma operator has no effect}} */
  (void)x;
}

void PotentiallyEvaluatedArrayBoundWarn(int n) {
  (void)*(int(*)[(0 << 32,n)])0; // expected-warning {{left operand of comma operator has no effect}}
}
