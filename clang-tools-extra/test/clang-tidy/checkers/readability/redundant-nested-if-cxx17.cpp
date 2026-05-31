// RUN: %check_clang_tidy -std=c++17-or-later %s readability-redundant-nested-if %t -- -- \
// RUN:   -I %S -fno-delayed-template-parsing

#include "Inputs/redundant-nested-if/common.h"

void init_statement_cases() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (side_effect(); cond()) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1))
      sink();
  }
  // CHECK-FIXES: if (side_effect(); (cond()) && (cond(1)))
  // CHECK-FIXES: sink();

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (bool B = cond()) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1))
      sink();
  }
  // CHECK-FIXES: if (bool B = cond(); B && (cond(1)))
  // CHECK-FIXES: sink();

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (bool X = cond(); X) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond()) {
      // CHECK-MESSAGES: :[[@LINE+1]]:7: note: nested 'if' statement to merge declared here
      if (cond(1))
        bar();
    }
  }
  // CHECK-FIXES: if (bool X = cond(); (X) && (cond()) && (cond(1)))
  // CHECK-FIXES: bar();

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (bool X = cond() /* here */) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond())
      bar();
  }
  // CHECK-FIXES: if (bool X = cond() /* here */; X && (cond()))
  // CHECK-FIXES: bar();
}

void declaration_condition_boollike_default_cases() {
  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (auto Guard = make_bool_like()) {
    if (cond(1))
      sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (bool X = COND_MACRO) {
    if (cond(1))
      sink();
  }
}

constexpr bool C1 = true;
constexpr bool C2 = false;

void constexpr_non_template() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (C1) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (C2)
      sink();
  }
  // CHECK-FIXES: if constexpr ((C1) && (C2))
  // CHECK-FIXES: sink();
}

template <bool B> void dependent_constexpr_outer_is_fixable() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (B) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (true)
      sink();
  }
  // CHECK-FIXES: if constexpr ((B) && (true))
  // CHECK-FIXES: sink();
}

template <bool B> void dependent_constexpr_outer_with_nested_false_is_fixable() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (B) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (false)
      sink();
  }
  // CHECK-FIXES: if constexpr ((B) && (false))
  // CHECK-FIXES: sink();
}

template <bool B, bool C> void dependent_constexpr_bool_operands_are_fixable() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (B) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (C)
      sink();
  }
  // CHECK-FIXES: if constexpr ((B) && (C))
  // CHECK-FIXES: sink();
}

template <typename T> void dependent_constexpr_operand_after_true_is_fixable() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (true) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (sizeof(T) == 4)
      sink();
  }
  // CHECK-FIXES: if constexpr ((true) && (sizeof(T) == 4))
  // CHECK-FIXES: sink();
}

template <bool B, typename T>
void dependent_constexpr_operand_after_dependent_is_unsafe() {
  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (B) {
    if constexpr (sizeof(typename T::type) == 4)
      sink();
  }
}

template <typename T> void dependent_constexpr_type_chain_outer_is_fixable() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (sizeof(T) == 4) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (true)
      sink();
  }
  // CHECK-FIXES: if constexpr ((sizeof(T) == 4) && (true))
  // CHECK-FIXES: sink();
}

void mixed_constexpr_and_non_constexpr(bool B) {
  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (C1) {
    if (B)
      sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (B) {
    if constexpr (C1)
      sink();
  }
}
