// RUN: %check_clang_tidy -check-suffixes=BASE -std=c++11 %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17 -std=c++17 %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17,CXX20 -std=c++20 %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17,CXX20,CXX23 -std=c++23 %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17,CXX20,CXX23,CXX26 -std=c++26 %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17,CXX20,CXX23,CXX26,OPTWARN -std=c++26 %s readability-redundant-nested-if %t -- -config='{CheckOptions: {readability-redundant-nested-if.WarnOnDependentConstexprIf: true, readability-redundant-nested-if.UserDefinedBoolConversionMode: WarnOnly}}' -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17,OPTFIX -std=c++17 %s readability-redundant-nested-if %t -- -config='{CheckOptions: {readability-redundant-nested-if.UserDefinedBoolConversionMode: WarnAndFix}}' -- -fno-delayed-template-parsing

bool cond(int X = 0);
void sink();
void bar();
struct BoolLike {
  explicit operator bool() const;
};
BoolLike make_bool_like();

#define INNER_IF(C) if (C) sink()
#define COND_MACRO cond()
#define OUTER_IF if (cond())

// Core coverage under default options.
void positive_cases() {
  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    if (cond(1)) {
      sink();
    }
  }
  // CHECK-FIXES-BASE: if ((cond()) && (cond(1)))
  // CHECK-FIXES-BASE: sink();

  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond() || cond(1))
    if (cond(2))
      sink();
  // CHECK-FIXES-BASE: if ((cond() || cond(1)) && (cond(2)))
  // CHECK-FIXES-BASE: sink();

  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    if (cond(1))
      if (cond(2)) {
        sink();
      }
  }
  // CHECK-FIXES-BASE: if ((cond()) && (cond(1)) && (cond(2)))
  // CHECK-FIXES-BASE: sink();
}

void stress_long_chain_case() {
  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond(0)) {
    if (cond(1)) {
      if (cond(2)) {
        if (cond(3)) {
          if (cond(4)) {
            if (cond(5)) {
              if (cond(6)) {
                if (cond(7)) {
                  if (cond(8)) {
                    if (cond(9)) {
                      if (cond(10)) {
                        if (cond(11))
                          sink();
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // CHECK-FIXES-BASE: if ((cond(0)) && (cond(1)) && (cond(2)) && (cond(3)) && (cond(4)) && (cond(5)) && (cond(6)) && (cond(7)) && (cond(8)) && (cond(9)) && (cond(10)) && (cond(11)))
  // CHECK-FIXES-BASE: sink();
}

void nested_chains_are_diagnosed_once_per_chain() {
  // CHECK-MESSAGES-BASE: :[[@LINE+2]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-BASE: :[[@LINE+4]]:7: warning: nested if statements can be merged
  if (cond()) {
    if (cond(1)) {
      sink();
      if (cond(2)) {
        if (cond(3))
          sink();
      }
    }
  }
  // CHECK-FIXES-BASE: if ((cond()) && (cond(1)))
  // CHECK-FIXES-BASE: if ((cond(2)) && (cond(3)))
}

void child_chain_is_reported_when_macro_parent_is_unfixable() {
  // CHECK-MESSAGES-BASE: :[[@LINE+2]]:5: warning: nested if statements can be merged
  OUTER_IF {
    if (cond(1)) {
      if (cond(2))
        sink();
    }
  }
  // CHECK-FIXES-BASE: OUTER_IF {
  // CHECK-FIXES-BASE: if ((cond(1)) && (cond(2)))
  // CHECK-FIXES-BASE: sink();
}

void negative_cases() {
  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (bool B = cond()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-CXX17: if (bool B = cond(); B && (cond(1)))
  // CHECK-FIXES-CXX17: sink();

  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    if (bool B = cond(1))
      sink();
  }

  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    if (cond(1))
      sink();
    else
      sink();
  }

  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    if (cond(1))
      sink();
  } else {
    sink();
  }

  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    sink();
    if (cond(1))
      sink();
  }

  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    if (cond(1))
      sink();
    sink();
  }

}

// Option-specific behavior for UserDefinedBoolConversionMode.
void user_defined_bool_conversion_mode_cases() {
  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+3]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-OPTWARN: :[[@LINE+2]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-OPTFIX: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (make_bool_like()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-OPTFIX: if ((make_bool_like()) && (cond(1)))
  // CHECK-FIXES-OPTFIX: sink();
}

void macro_and_preprocessor_cases() {
  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    INNER_IF(cond(1));
  }

  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (COND_MACRO) {
    if (cond(1))
      sink();
  }

  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
#if 1
    if (cond(1))
      sink();
#endif
  }
}

void comment_handling_cases() {
  // Comments inside condition payloads are preserved by the merged condition.
  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond() /* outer payload */) {
    if (/* inner payload */ cond(1))
      sink();
  }
  // CHECK-FIXES-BASE: if ((cond() /* outer payload */) && (/* inner payload */ cond(1)))
  // CHECK-FIXES-BASE: sink();

  // Trailing comments in nested headers are warning-only (no fix-it).
  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) // outer trailing
    if (cond(1)) // inner trailing
      sink();
  // CHECK-FIXES-BASE: if (cond()) // outer trailing
  // CHECK-FIXES-BASE-NEXT: if (cond(1)) // inner trailing
  // CHECK-FIXES-BASE-NEXT: sink();

  // Comments in other nested-header locations are warning-only (no fix-it).
  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    if /* nested header comment */ (cond(1))
      sink();
  }
  // CHECK-FIXES-BASE: if /* nested header comment */ (cond(1))
}

#if __cplusplus >= 201703L
int side_effect();

// C++17 language feature coverage.
void init_statement_cases() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (side_effect(); cond()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-CXX17: if (side_effect(); (cond()) && (cond(1)))
  // CHECK-FIXES-CXX17: sink();

  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (auto Guard = side_effect()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-CXX17: if (auto Guard = side_effect(); Guard && (cond(1)))
  // CHECK-FIXES-CXX17: sink();

  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    if (bool InnerInit = cond(1); InnerInit)
      sink();
  }

  // Macro-expanded root conditions are diagnostic-only unsafe, so no warning
  // with fix-it is emitted.
  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (side_effect(); COND_MACRO) {
    if (cond(1))
      sink();
  }

  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (bool X = cond(); X) {
    if (cond()) {
      if (cond(1))
        bar();
    }
  }
  // CHECK-FIXES-CXX17: if (bool X = cond(); (X) && (cond()) && (cond(1)))
  // CHECK-FIXES-CXX17: bar();

  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (bool X = cond() /* here */) {
    if (cond()) {
      bar();
    }
  }
  // CHECK-FIXES-CXX17: if (bool X = cond() /* here */; X && (cond()))
  // CHECK-FIXES-CXX17: bar();
}

void declaration_condition_type_safety() {
  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+3]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-OPTWARN: :[[@LINE+2]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-OPTFIX: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (auto Guard = make_bool_like()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-OPTFIX: if (auto Guard = make_bool_like(); Guard && (cond(1)))
  // CHECK-FIXES-OPTFIX: sink();

  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (bool X = cond()) {
    if (cond()) {
      if (cond(1))
        bar();
    }
  }
  // CHECK-FIXES-CXX17: if (bool X = cond(); X && (cond()) && (cond(1)))
  // CHECK-FIXES-CXX17: bar();

  // Macro-expanded declaration conditions are diagnostic-only unsafe, so no
  // warning with fix-it is emitted.
  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (bool X = COND_MACRO) {
    if (cond(1))
      sink();
  }

}

constexpr bool C1 = true;
constexpr bool C2 = false;

void constexpr_non_template() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (C1) {
    if constexpr (C2) {
      sink();
    }
  }
  // CHECK-FIXES-CXX17: if constexpr ((C1) && (C2))
  // CHECK-FIXES-CXX17: sink();
}

template <bool B> void dependent_constexpr_outer_is_fixable() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (B) {
    if constexpr (true)
      sink();
  }
  // CHECK-FIXES-CXX17: if constexpr ((B) && (true))
  // CHECK-FIXES-CXX17: sink();
}

template <bool B> void dependent_constexpr_outer_is_unsafe_when_nested_false() {
  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-OPTWARN: :[[@LINE+1]]:3: warning: nested instantiation-dependent if constexpr statements can be merged
  if constexpr (B) {
    if constexpr (false)
      sink();
  }
}

template <typename T> void dependent_constexpr_operand_warn_only_under_option() {
  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-OPTWARN: :[[@LINE+1]]:3: warning: nested instantiation-dependent if constexpr statements can be merged
  if constexpr (true) {
    if constexpr (sizeof(T) == 4)
      sink();
  }
}

template <typename T> void dependent_constexpr_type_chain_outer_is_fixable() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (sizeof(T) == 4) {
    if constexpr (true)
      sink();
  }
  // CHECK-FIXES-CXX17: if constexpr ((sizeof(T) == 4) && (true))
  // CHECK-FIXES-CXX17: sink();
}

void mixed_constexpr_and_non_constexpr(bool B) {
  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (C1) {
    if (B)
      sink();
  }

  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (B) {
    if constexpr (C1)
      sink();
  }
}

#if __cplusplus >= 202400L
template <typename T> void constexpr_template_static_assert() {
  // CHECK-MESSAGES-CXX26-NOT: :[[@LINE+2]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-OPTWARN: :[[@LINE+1]]:3: warning: nested instantiation-dependent if constexpr statements can be merged
  if constexpr (sizeof(T) == 1) {
    if constexpr (false) {
      static_assert(false, "discarded in template context");
    }
  }
}
#endif

#endif

#if __cplusplus >= 202002L
void constexpr_requires_expression_cases() {
  // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (requires { sizeof(int); }) {
    if constexpr (requires { sizeof(long); })
      sink();
  }
  // CHECK-FIXES-CXX20: if constexpr ((requires { sizeof(int); }) && (requires { sizeof(long); }))
  // CHECK-FIXES-CXX20: sink();
}

void constexpr_init_statement_cases() {
  // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (constexpr bool HasInt = requires { sizeof(int); }; HasInt) {
    if constexpr (requires { sizeof(long); })
      sink();
  }
  // CHECK-FIXES-CXX20: if constexpr (constexpr bool HasInt = requires { sizeof(int); }; (HasInt) && (requires { sizeof(long); }))
  // CHECK-FIXES-CXX20: sink();
}

template <typename T> void dependent_requires_outer_is_fixable() {
  // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (requires { typename T::type; }) {
    if constexpr (true)
      sink();
  }
  // CHECK-FIXES-CXX20: if constexpr ((requires { typename T::type; }) && (true))
  // CHECK-FIXES-CXX20: sink();
}

void attribute_cases(bool B1, bool B2) {
  // CHECK-MESSAGES-CXX20-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (B1) {
    // CHECK-MESSAGES-CXX20-NOT: :[[@LINE+1]]:5: warning: nested if statements can be merged
    [[likely]] if (B2)
      sink();
  }

  // CHECK-MESSAGES-CXX20-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  [[likely]] if (B1) {
    if (B2)
      sink();
  }
}

void still_merges_without_attributes(bool B1, bool B2) {
  // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (B1) {
    if (B2)
      sink();
  }
  // CHECK-FIXES-CXX20: if ((B1) && (B2))
  // CHECK-FIXES-CXX20: sink();
}
#endif

#ifdef __cpp_if_consteval
consteval bool compile_time_true() { return true; }

void consteval_cases(bool B1, bool B2) {
  // CHECK-MESSAGES-CXX23-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if consteval {
    if (B1)
      sink();
  } else {
    sink();
  }

  // CHECK-MESSAGES-CXX23-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (!B1) {
    if consteval {
      sink();
    } else {
      if (B2)
        sink();
    }
  }

}

template <bool B> constexpr void consteval_nested_in_constexpr() {
  // CHECK-MESSAGES-CXX23-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (B) {
    if consteval {
      sink();
    }
  }
}
#endif
