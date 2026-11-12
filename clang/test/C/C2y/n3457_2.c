// RUN: %clang_cc1 -verify=good -std=c2y -finitial-counter-value=2147483648 %s
// RUN: %clang_cc1 -verify -std=c2y -finitial-counter-value=2147483648 -DEXPAND_IT %s
// good-no-diagnostics

// This sets the intial __COUNTER__ value to something that's too big. Setting
// the value too large is fine. Expanding to a too-large value is not.
#ifdef EXPAND_IT
  // This one should fail.
  signed long i = __COUNTER__; // expected-error {{'__COUNTER__' value cannot exceed 2'147'483'647}}
#endif
