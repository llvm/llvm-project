// RUN: %check_clang_tidy %s readability-enum-initial-value %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-enum-initial-value.AllowConsecutiveInitialValuesExceptLast: true, \
// RUN:         readability-enum-initial-value.AllowReferencedInitialValues: true, \
// RUN:     }}'

// OK: an alias of the first enumerator.
enum EAlias {
  EAlias_a = 4,
  EAlias_b = EAlias_a,
  EAlias_size,
};

// OK: an alias in a class template.
template <int V>
struct TemplateAlias {
  enum E { A = V, B = A, Size };
};

// Error: aliases are not skipped by AllowConsecutiveInitialValuesExceptLast.
enum EConsecutiveAlias {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EConsecutiveAlias' are not consistent
  EConsecutiveAlias_a = 0,
  EConsecutiveAlias_b = 1,
  EConsecutiveAlias_first = EConsecutiveAlias_a,
  EConsecutiveAlias_c = 2,
  EConsecutiveAlias_size,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EConsecutiveAlias_size' defined here
  // CHECK-FIXES: EConsecutiveAlias_size = 3,
};

// Error: aliases are not skipped by AllowConsecutiveInitialValuesExceptLast,
// in a class template.
template <int V>
struct TemplateConsecutiveAlias {
  enum E {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: initial values in enum 'TemplateConsecutiveAlias::E' are not consistent
    A = V,
    B = V + 1,
    First = A,
    C = V + 2,
    Size,
    // CHECK-MESSAGES: :[[@LINE-1]]:5: note: uninitialized enumerator 'Size' defined here
  };
};

// OK: referencing the previous enumerator in a class template.
template <int V>
struct TemplateReferenceAliasWithImplicitTrailing {
  enum E {
    A = V,
    B = A,
    Size,
    Last,
  };
};

// Error: consecutive values are allowed except for the non-final implicit
// enumerator.
template <int V>
struct TemplateConsecutiveWithImplicitNonTrailing {
  enum E {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: initial values in enum 'TemplateConsecutiveWithImplicitNonTrailing::E' are not consistent
    A = V,
    B = A + 1,
    Size,
    // CHECK-MESSAGES: :[[@LINE-1]]:5: note: uninitialized enumerator 'Size' defined here
    Last,
  };
};

// Error: an alias before the last enumerator changes its implicit value.
enum EAliasBeforeLast {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EAliasBeforeLast' are not consistent
  EAliasBeforeLast_a = 0,
  EAliasBeforeLast_b = 1,
  EAliasBeforeLast_first = EAliasBeforeLast_a,
  EAliasBeforeLast_size,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EAliasBeforeLast_size' defined here
  // CHECK-FIXES: EAliasBeforeLast_size = 1,
};

// Error: an alias before the last enumerator changes its implicit value, in a
// class template.
template <int V>
struct TemplateAliasBeforeLast {
  enum E {
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: initial values in enum 'TemplateAliasBeforeLast::E' are not consistent
    A = V,
    B = V + 1,
    First = A,
    Size,
    // CHECK-MESSAGES: :[[@LINE-1]]:5: note: uninitialized enumerator 'Size' defined here
  };
};

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
