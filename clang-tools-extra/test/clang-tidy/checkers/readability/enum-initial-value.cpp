// RUN: %check_clang_tidy %s readability-enum-initial-value %t

enum class EError {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EError' are not consistent, consider explicit initialization of all, none or only the first enumerator
  EError_a = 1,
  EError_b,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EError_b' defined here
  // CHECK-FIXES: EError_b = 2,
  EError_c = 3,
};

enum class ENone {
  ENone_a,
  ENone_b,
  EENone_c,
};

enum class EFirst {
  EFirst_a = 1,
  EFirst_b,
  EFirst_c,
};

enum class EAll {
  EAll_a = 1,
  EAll_b = 2,
  EAll_c = 3,
};

enum class ERef {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'ERef' are not consistent
  ERef_a,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'ERef_a' defined here
  // CHECK-FIXES: ERef_a = 0,
  ERef_b,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'ERef_b' defined here
  // CHECK-FIXES: ERef_b = 1,
  ERef_last = ERef_b,
};

enum class EConsecutiveExceptLast {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EConsecutiveExceptLast' are not consistent
  EConsecutiveExceptLast_a = 10,
  EConsecutiveExceptLast_b = 11,
  EConsecutiveExceptLast_c = 12,
  EConsecutiveExceptLast_size,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EConsecutiveExceptLast_size' defined here
  // CHECK-FIXES: EConsecutiveExceptLast_size = 13,
};

// OK: all enumerators initialized in a class template.
template <typename T, T V>
struct TemplateAllInitialized {
  enum E { A = V, B = V + 1, C = V + 2 };
};
template struct TemplateAllInitialized<int, 3>;

// OK: only the first enumerator initialized in a class template.
template <typename T, T V>
struct TemplateOnlyFirstInitialized {
  enum E { A = V, B, C };
};
template struct TemplateOnlyFirstInitialized<int, 3>;

// Error: consecutive except last is not allowed by default, also in templates.
template <typename T, T V>
struct TemplateConsecutiveExceptLast {
  enum E {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: initial values in enum 'TemplateConsecutiveExceptLast::E' are not consistent
    A = V,
    B = V + 1,
    Size,
    // CHECK-MESSAGES: :[[@LINE-1]]:5: note: uninitialized enumerator 'Size' defined here
  };
};

