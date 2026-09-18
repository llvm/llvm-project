// RUN: %check_clang_tidy %s readability-enum-initial-value %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-enum-initial-value.AllowReferencedInitialValues: true, \
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

// OK: first-only + self-refs, where later self-references do not need to
// immediately follow the enumerator they reference and do not require every
// preceding enumerator to be explicitly initialized.
enum EFirstThenRefs {
  EFirstThenRefs_a = 0,
  EFirstThenRefs_b,
  EFirstThenRefs_c,
  EFirstThenRefs_first = EFirstThenRefs_a,
  EFirstThenRefs_last = EFirstThenRefs_c,
};

// OK: a self-reference may appear before later implicit enumerators.
enum ERefThenImplicit {
  ERefThenImplicit_a = 0,
  ERefThenImplicit_alias = ERefThenImplicit_a,
  ERefThenImplicit_b,
  ERefThenImplicit_c,
};

// Error: literal duplicate (not a reference).
enum class ERefErr {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'ERefErr' are not consistent
  ERefErr_a,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'ERefErr_a' defined here
  // CHECK-FIXES: ERefErr_a = 0,
  ERefErr_b,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'ERefErr_b' defined here
  // CHECK-FIXES: ERefErr_b = 1,
  ERef_last = 1,
};
