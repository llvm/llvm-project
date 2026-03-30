// RUN: %check_clang_tidy %s readability-enum-initial-value %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-enum-initial-value.AllowExplicitReferencedInitialValues: true, \
// RUN:     }}'

// OK: none + self-ref.
enum ERef1 {
  ERef1_a,
  ERef1_b,
  ERef1_last = ERef1_b,
};

// OK: first-only + self-ref.
enum ERef2 {
  ERef2_a = 0,
  ERef2_b,
  ERef2_last = ERef2_b,
};

// OK: all + self-ref.
enum ERef3 {
  ERef3_a = 0,
  ERef3_b = 1,
  ERef3_last = ERef3_b,
};

// Error: cross-enum reference is not a self-ref.
enum ERefOther {
  ERefOther_a,
  ERefOther_b,
};
enum ERefCross {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'ERefCross' are not consistent
  ERefCross_a,
  // CHECK-FIXES: ERefCross_a = 0,
  ERefCross_b,
  // CHECK-FIXES: ERefCross_b = 1,
  ERefCross_last = ERefOther_b,
};

// Error: inconsistent even after ignoring self-refs.
enum ERefBad {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'ERefBad' are not consistent
  ERefBad_a = 1,
  ERefBad_b,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'ERefBad_b' defined here
  // CHECK-FIXES: ERefBad_b = 2,
  ERefBad_c = 5,
  ERefBad_alias = ERefBad_a,
};
