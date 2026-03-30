// RUN: %check_clang_tidy %s readability-enum-initial-value %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-enum-initial-value.AllowExplicitZeroFirstInitialValue: false, \
// RUN:         readability-enum-initial-value.AllowExplicitSequentialInitialValues: false, \
// RUN:         readability-enum-initial-value.AllowExplicitReferencedInitialValues: true, \
// RUN:     }}'

// Error: zero-first + self-ref, should still warn about the zero.
enum EZeroRef {
  EZeroRef_a = 0,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: zero initial value for the first enumerator in 'EZeroRef' can be disregarded
  // CHECK-FIXES: EZeroRef_a ,
  EZeroRef_b,
  EZeroRef_last = EZeroRef_b,
};

// Error: sequential + self-ref, should still warn but not suggest
// removing the self-ref.
enum ESeqRef {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: sequential initial value in 'ESeqRef' can be ignored
  ESeqRef_a = 1,
  ESeqRef_b = 2,
  // CHECK-FIXES: ESeqRef_b ,
  ESeqRef_c = 3,
  // CHECK-FIXES: ESeqRef_c ,
  ESeqRef_alias = ESeqRef_a,
};

// OK: none + self-ref, no warnings.
enum ENoneRef {
  ENoneRef_a,
  ENoneRef_b,
  ENoneRef_last = ENoneRef_b,
};
