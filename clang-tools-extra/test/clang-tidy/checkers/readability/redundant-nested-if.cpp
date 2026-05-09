// Use exact standards because this file has version-specific FileCheck
// suffixes. `check_clang_tidy.py` expands `-std=c++17-or-later` into separate
// C++17, C++20, ... runs, but reuses the same suffix list for each run. A
// suffix list for one language mode would either miss warnings from later
// `#if __cplusplus` blocks, or look for CHECK lines from blocks preprocessed
// away in earlier modes.
// RUN: %check_clang_tidy -check-suffixes=BASE -std=c++98,c++11,c++14 %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17 -std=c++17 %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17,CXX20 -std=c++20 %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17,CXX20,CXX23 -std=c++23 %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17,CXX20,CXX23,CXX26 -std=c++26-or-later %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffixes=BASE,CXX17,CXX20,CXX23,CXX26,ALLOWBOOL -std=c++26-or-later %s readability-redundant-nested-if %t -- -config='{CheckOptions: {readability-redundant-nested-if.AllowUserDefinedBoolConversion: true}}' -- -fno-delayed-template-parsing

bool cond(int X = 0);
int side_effect();
void sink();
void bar();

struct BoolLike {
  operator bool() const;
};

BoolLike make_bool_like();

#define INNER_IF(C) if (C) sink()
#define COND_MACRO cond()
#define OUTER_IF if (cond())

void positive_cases() {
  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    // CHECK-MESSAGES-BASE: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (cond(1)) {
      sink();
    }
  }
  // CHECK-FIXES-BASE: if ((cond()) && (cond(1)))
  // CHECK-FIXES-BASE: sink();

  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    // CHECK-MESSAGES-BASE: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (cond(1)) {
      // CHECK-MESSAGES-BASE: :[[@LINE+1]]:7: note: nested if statement to merge is here
      if (cond(2))
        sink();
    }
  }
  // CHECK-FIXES-BASE: if ((cond()) && (cond(1)) && (cond(2)))
  // CHECK-FIXES-BASE: sink();

  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond() || cond(1))
    // CHECK-MESSAGES-BASE: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (cond(2))
      sink();
  // CHECK-FIXES-BASE: if ((cond() || cond(1)) && (cond(2)))
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
  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    // CHECK-MESSAGES-BASE: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (cond(1)) {
      sink();
      // CHECK-MESSAGES-BASE: :[[@LINE+1]]:7: warning: nested if statements can be merged
      if (cond(2)) {
        // CHECK-MESSAGES-BASE: :[[@LINE+1]]:9: note: nested if statement to merge is here
        if (cond(3))
          sink();
      }
    }
  }
  // CHECK-FIXES-BASE: if ((cond()) && (cond(1)))
  // CHECK-FIXES-BASE: if ((cond(2)) && (cond(3)))
}

void child_chain_is_reported_when_parent_is_not_diagnosable() {
  // CHECK-MESSAGES-BASE: :[[@LINE+2]]:5: warning: nested if statements can be merged
  OUTER_IF {
    if (cond(1)) {
      // CHECK-MESSAGES-BASE: :[[@LINE+1]]:7: note: nested if statement to merge is here
      if (cond(2))
        sink();
    }
  }
  // CHECK-FIXES-BASE: OUTER_IF {
  // CHECK-FIXES-BASE: if ((cond(1)) && (cond(2)))
  // CHECK-FIXES-BASE: sink();
}

void else_branch_child_chain_is_reported_when_parent_is_not_diagnosable() {
  // CHECK-MESSAGES-BASE: :[[@LINE+4]]:5: warning: nested if statements can be merged
  if (cond()) {
    sink();
  } else {
    if (cond(1)) {
      // CHECK-MESSAGES-BASE: :[[@LINE+1]]:7: note: nested if statement to merge is here
      if (cond(2))
        sink();
    }
  }
  // CHECK-FIXES-BASE: } else {
  // CHECK-FIXES-BASE: if ((cond(1)) && (cond(2)))
  // CHECK-FIXES-BASE: sink();
}

void negative_cases() {
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
    // CHECK-MESSAGES-BASE: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (/* inner payload */ cond(1))
      sink();
  }
  // CHECK-FIXES-BASE: if ((cond() /* outer payload */) && (/* inner payload */ cond(1)))
  // CHECK-FIXES-BASE: sink();

  // Trailing comments in nested headers keep the diagnostic but suppress fix-its.
  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) // outer trailing
    // CHECK-MESSAGES-BASE: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (cond(1)) // inner trailing
      sink();
  // CHECK-FIXES-BASE: if (cond()) // outer trailing
  // CHECK-FIXES-BASE: if (cond(1)) // inner trailing
  // CHECK-FIXES-BASE: sink();

  // CHECK-MESSAGES-BASE: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (cond()) {
    // CHECK-MESSAGES-BASE: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if /* nested header comment */ (cond(1))
      sink();
  }
  // CHECK-FIXES-BASE: if /* nested header comment */ (cond(1))
}

void user_defined_bool_conversion_cases() {
  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+3]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:5: note: nested if statement to merge is here
  if (make_bool_like()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-ALLOWBOOL: if ((static_cast<bool>(make_bool_like())) && (cond(1)))
  // CHECK-FIXES-ALLOWBOOL: sink();

  // CHECK-MESSAGES-BASE-NOT: :[[@LINE+3]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:5: note: nested if statement to merge is here
  if (cond(1)) {
    if (make_bool_like())
      sink();
  }
  // CHECK-FIXES-ALLOWBOOL: if ((cond(1)) && (static_cast<bool>(make_bool_like())))
  // CHECK-FIXES-ALLOWBOOL: sink();
}

#if __cplusplus >= 201703L
void init_statement_cases() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (side_effect(); cond()) {
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-CXX17: if (side_effect(); (cond()) && (cond(1)))
  // CHECK-FIXES-CXX17: sink();

  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (bool B = cond()) {
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-CXX17: if (bool B = cond(); B && (cond(1)))
  // CHECK-FIXES-CXX17: sink();

  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (bool X = cond(); X) {
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (cond()) {
      // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:7: note: nested if statement to merge is here
      if (cond(1))
        bar();
    }
  }
  // CHECK-FIXES-CXX17: if (bool X = cond(); (X) && (cond()) && (cond(1)))
  // CHECK-FIXES-CXX17: bar();

  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (bool X = cond() /* here */) {
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (cond())
      bar();
  }
  // CHECK-FIXES-CXX17: if (bool X = cond() /* here */; X && (cond()))
  // CHECK-FIXES-CXX17: bar();
}

void declaration_condition_boollike_cases() {
  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+3]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:3: warning: nested if statements can be merged
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:5: note: nested if statement to merge is here
  if (auto Guard = make_bool_like()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-ALLOWBOOL: if (auto Guard = make_bool_like(); static_cast<bool>(Guard) && (cond(1)))
  // CHECK-FIXES-ALLOWBOOL: sink();

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
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (C2)
      sink();
  }
  // CHECK-FIXES-CXX17: if constexpr ((C1) && (C2))
  // CHECK-FIXES-CXX17: sink();
}

template <bool B> void dependent_constexpr_outer_is_fixable() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (B) {
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (true)
      sink();
  }
  // CHECK-FIXES-CXX17: if constexpr ((B) && (true))
  // CHECK-FIXES-CXX17: sink();
}

template <bool B> void dependent_constexpr_outer_with_nested_false_is_fixable() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (B) {
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (false)
      sink();
  }
  // CHECK-FIXES-CXX17: if constexpr ((B) && (false))
  // CHECK-FIXES-CXX17: sink();
}

template <bool B, bool C> void dependent_constexpr_bool_operands_are_fixable() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (B) {
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (C)
      sink();
  }
  // CHECK-FIXES-CXX17: if constexpr ((B) && (C))
  // CHECK-FIXES-CXX17: sink();
}

template <typename T> void dependent_constexpr_operand_after_true_is_fixable() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (true) {
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (sizeof(T) == 4)
      sink();
  }
  // CHECK-FIXES-CXX17: if constexpr ((true) && (sizeof(T) == 4))
  // CHECK-FIXES-CXX17: sink();
}

template <bool B, typename T>
void dependent_constexpr_operand_after_dependent_is_unsafe() {
  // CHECK-MESSAGES-CXX17-NOT: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (B) {
    if constexpr (sizeof(typename T::type) == 4)
      sink();
  }
}

template <typename T> void dependent_constexpr_type_chain_outer_is_fixable() {
  // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (sizeof(T) == 4) {
    // CHECK-MESSAGES-CXX17: :[[@LINE+1]]:5: note: nested if statement to merge is here
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
  // CHECK-MESSAGES-CXX26: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (sizeof(T) == 1) {
    // CHECK-MESSAGES-CXX26: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (false) {
      static_assert(false, "discarded in template context");
    }
  }
  // CHECK-FIXES-CXX26: if constexpr ((sizeof(T) == 1) && (false))
  // CHECK-FIXES-CXX26: static_assert(false, "discarded in template context");
}
#endif

#endif

#if __cplusplus >= 202002L
void constexpr_requires_expression_cases() {
  // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (requires { sizeof(int); }) {
    // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (requires { sizeof(long); })
      sink();
  }
  // CHECK-FIXES-CXX20: if constexpr ((requires { sizeof(int); }) && (requires { sizeof(long); }))
  // CHECK-FIXES-CXX20: sink();
}

void constexpr_init_statement_cases() {
  // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (constexpr bool HasInt = requires { sizeof(int); }; HasInt) {
    // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (requires { sizeof(long); })
      sink();
  }
  // CHECK-FIXES-CXX20: if constexpr (constexpr bool HasInt = requires { sizeof(int); }; (HasInt) && (requires { sizeof(long); }))
  // CHECK-FIXES-CXX20: sink();
}

template <typename T> void dependent_requires_outer_is_fixable() {
  // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (requires { typename T::type; }) {
    // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (true)
      sink();
  }
  // CHECK-FIXES-CXX20: if constexpr ((requires { typename T::type; }) && (true))
  // CHECK-FIXES-CXX20: sink();
}

template <bool B, typename T> void dependent_requires_after_bool_is_fixable() {
  // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if constexpr (B) {
    // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if constexpr (requires { typename T::type; })
      sink();
  }
  // CHECK-FIXES-CXX20: if constexpr ((B) && (requires { typename T::type; }))
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

  // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:3: warning: nested if statements can be merged
  if (B1) {
    // CHECK-MESSAGES-CXX20: :[[@LINE+1]]:5: note: nested if statement to merge is here
    if (B2)
      sink();
  }
  // CHECK-FIXES-CXX20: if ((B1) && (B2))
  // CHECK-FIXES-CXX20: sink();
}
#endif

#ifdef __cpp_if_consteval
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
