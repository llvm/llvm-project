// RUN: %check_clang_tidy %s readability-enum-initial-value %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-enum-initial-value.AllowExplicitZeroFirstInitialValue: false, \
// RUN:         readability-enum-initial-value.AllowExplicitSequentialInitialValues: false, \
// RUN:         readability-enum-initial-value.AllowReferencedInitialValues: true, \
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

// Error: sequential + multiple self-refs, should still warn and suggest
// removing the sequential values but not the self-refs.
enum ESeqMultiRef {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: sequential initial value in 'ESeqMultiRef' can be ignored
  ESeqMultiRef_a = 1,
  ESeqMultiRef_b = 2,
  // CHECK-FIXES: ESeqMultiRef_b ,
  ESeqMultiRef_c = 3,
  // CHECK-FIXES: ESeqMultiRef_c ,
  ESeqMultiRef_alias = ESeqMultiRef_a,
  ESeqMultiRef_alias2 = ESeqMultiRef_b,
};

// Error: sequential + self-refs interleaved with the sequence, should still
// warn and suggest removing the sequential values but not the self-refs.
enum ESeqInterRef {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: sequential initial value in 'ESeqInterRef' can be ignored
  ESeqInterRef_a = 1,
  ESeqInterRef_alias = ESeqInterRef_a,
  ESeqInterRef_b = 2,
  // CHECK-FIXES: ESeqInterRef_b ,
  ESeqInterRef_c = 3,
  // CHECK-FIXES: ESeqInterRef_c ,
  ESeqInterRef_alias2 = ESeqInterRef_b,
};

// Error: sequential, but an enumerator immediately follows a run of self-refs
// whose value breaks the natural progression. Removing its explicit value
// would silently change it, so it must be kept.
enum ESeqRefBreak {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: sequential initial value in 'ESeqRefBreak' can be ignored
  ESeqRefBreak_a = 1,
  ESeqRefBreak_alias = ESeqRefBreak_a,
  ESeqRefBreak_b = 2,
  // CHECK-FIXES: ESeqRefBreak_b ,
  ESeqRefBreak_c = 3,
  // CHECK-FIXES: ESeqRefBreak_c ,
  ESeqRefBreak_alias2 = ESeqRefBreak_b,
  ESeqRefBreak_alias3 = ESeqRefBreak_a,
  ESeqRefBreak_d = 4,
  // CHECK-FIXES: ESeqRefBreak_d = 4,
  ESeqRefBreak_e = 5,
  // CHECK-FIXES: ESeqRefBreak_e ,
};

// OK: none + self-ref, no warnings.
enum ENoneRef {
  ENoneRef_a,
  ENoneRef_b,
  ENoneRef_last = ENoneRef_b,
};
