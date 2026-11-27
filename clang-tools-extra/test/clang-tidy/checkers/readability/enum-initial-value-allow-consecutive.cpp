// RUN: %check_clang_tidy %s readability-enum-initial-value %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-enum-initial-value.AllowConsecutiveInitialValuesExceptLast: true, \
// RUN:     }}'

// OK: consecutive except last, starting at zero.
enum class EConsecutive {
  EConsecutive_a = 0,
  EConsecutive_b = 1,
  EConsecutive_c = 2,
  EConsecutive_size,
};

// OK: consecutive except last, with a non-zero starting offset.
enum class EConsecutiveOffset {
  EConsecutiveOffset_a = 10,
  EConsecutiveOffset_b = 11,
  EConsecutiveOffset_c = 12,
  EConsecutiveOffset_size,
};

// OK: consecutive except last, with negative values.
enum class EConsecutiveNegative {
  EConsecutiveNegative_a = -2,
  EConsecutiveNegative_b = -1,
  EConsecutiveNegative_c = 0,
  EConsecutiveNegative_size,
};

// OK: minimal case, only two enumerators.
enum class EConsecutiveMinimal {
  EConsecutiveMinimal_a = 5,
  EConsecutiveMinimal_size,
};

// OK: constant expression initializers with consecutive values.
enum class EConsecutiveConstExpr {
  EConsecutiveConstExpr_a = 1 << 0,
  EConsecutiveConstExpr_b = 1 << 1,
  EConsecutiveConstExpr_size,
};

// Error: constant expression initializers with non-consecutive values.
enum class EConstExprBreak {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EConstExprBreak' are not consistent
  EConstExprBreak_a = 1 << 0,
  EConstExprBreak_b = 1 << 1,
  EConstExprBreak_c = 1 << 2,
  EConstExprBreak_size,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EConstExprBreak_size' defined here
  // CHECK-FIXES: EConstExprBreak_size = 5,
};

// Error: not consecutive (gap between b and c).
enum class EConsecutiveBreak {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EConsecutiveBreak' are not consistent
  EConsecutiveBreak_a = 0,
  EConsecutiveBreak_b = 1,
  EConsecutiveBreak_c = 3,
  EConsecutiveBreak_size,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EConsecutiveBreak_size' defined here
  // CHECK-FIXES: EConsecutiveBreak_size = 4,
};

// Error: an enumerator other than the last is uninitialized.
enum class EConsecutiveGap {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EConsecutiveGap' are not consistent
  EConsecutiveGap_a = 0,
  EConsecutiveGap_b,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EConsecutiveGap_b' defined here
  // CHECK-FIXES: EConsecutiveGap_b = 1,
  EConsecutiveGap_c = 2,
  EConsecutiveGap_size,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EConsecutiveGap_size' defined here
  // CHECK-FIXES: EConsecutiveGap_size = 3,
};

// Error: the last enumerator is explicitly initialized.
enum class EConsecutiveLastMismatch {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EConsecutiveLastMismatch' are not consistent
  EConsecutiveLastMismatch_a = 0,
  EConsecutiveLastMismatch_b = 1,
  EConsecutiveLastMismatch_c,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EConsecutiveLastMismatch_c' defined here
  // CHECK-FIXES: EConsecutiveLastMismatch_c = 2,
  EConsecutiveLastMismatch_last = 10,
};

// OK: consecutive except last inside a class template.
template <typename T>
struct TemplateConsecutive {
  enum E { A = 0, B = 1, Size };
};
template struct TemplateConsecutive<int>;

// OK: value-dependent initializers with consecutive offsets.
template <typename T, T V>
struct TemplateDependent {
  enum E { A = V, B = V + 1, Size };
};
template struct TemplateDependent<int, 3>;

// OK: value-dependent initializers with consecutive offsets, alternate forms.
template <int V>
struct TemplateDependentForms {
  enum E { A = V - 1, B = (V), C = 1 + V, D = C + 1, Size };
};
template struct TemplateDependentForms<5>;

// Error: the difference between unrelated template parameters is unknown.
template <int V, int W>
struct TemplateDependentUnrelated {
  enum E {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: initial values in enum 'TemplateDependentUnrelated::E' are not consistent
    A = V,
    B = W,
    Size,
    // CHECK-MESSAGES: :[[@LINE-1]]:5: note: uninitialized enumerator 'Size' defined here
  };
};

// Error: value-dependent initializers with non-consecutive offsets.
template <typename T, T V>
struct TemplateDependentBreak {
  enum E {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: initial values in enum 'TemplateDependentBreak::E' are not consistent
    A = V,
    B = V + 10,
    Size,
    // CHECK-MESSAGES: :[[@LINE-1]]:5: note: uninitialized enumerator 'Size' defined here
  };
};

// Error: non-dependent initializers with non-consecutive values in a template.
template <typename T>
struct TemplateNonDependentBreak {
  enum E {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: initial values in enum 'TemplateNonDependentBreak::E' are not consistent
    A = 0,
    B = 2,
    Size,
    // CHECK-MESSAGES: :[[@LINE-1]]:5: note: uninitialized enumerator 'Size' defined here
  };
};

// OK: referencing the previous enumerator with a consecutive offset.
template <int V>
struct TemplateReferenceConsecutive {
  enum E { A = V, B = A + 1, Size };
};

// Error: an alias is not consecutive, as in a non-template enum.
template <int V>
struct TemplateAlias {
  enum E {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: initial values in enum 'TemplateAlias::E' are not consistent
    A = V,
    B = A,
    Size,
    // CHECK-MESSAGES: :[[@LINE-1]]:5: note: uninitialized enumerator 'Size' defined here
  };
};

// Error: an alias is not consecutive, non-template counterpart.
enum EAlias {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EAlias' are not consistent
  EAlias_a = 4,
  EAlias_b = EAlias_a,
  EAlias_size,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EAlias_size' defined here
  // CHECK-FIXES: EAlias_size = 5,
};
