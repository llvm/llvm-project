// RUN: %check_clang_tidy %s readability-enum-initial-value %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-enum-initial-value.AllowExplicitReferencedInitialValues: true, \
// RUN:     }}'

// OK: none + self-ref.
enum class ERef {
  ERef_a,
  ERef_b,
  ERef_last = ERef_b,
};

// OK: first-only + self-ref.
enum class ERefFirst {
  ERefFirst_a = 1,
  ERefFirst_b,
  ERefFirst_alias = ERefFirst_a,
};

// OK: all + self-ref.
enum class ERefAll {
  ERefAll_a = 0,
  ERefAll_b = 1,
  ERefAll_last = ERefAll_b,
};

// Error: literal duplicate (not a reference).
enum class ERefErr {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'ERefErr' are not consistent
  ERefErr_a,
  // CHECK-FIXES: ERefErr_a = 0,
  ERefErr_b,
  // CHECK-FIXES: ERefErr_b = 1,
  ERef_last = 1,
};
