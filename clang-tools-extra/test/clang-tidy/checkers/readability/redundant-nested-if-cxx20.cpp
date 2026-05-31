// RUN: %check_clang_tidy -std=c++20-or-later %s readability-redundant-nested-if %t -- -- \
// RUN:   -I %S -fno-delayed-template-parsing

#include "Inputs/redundant-nested-if/common.h"

void constexpr_requires_expression_cases() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (requires { sizeof(int); }) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (requires { sizeof(long); })
      sink();
  }
  // CHECK-FIXES: if constexpr ((requires { sizeof(int); }) && (requires { sizeof(long); }))
  // CHECK-FIXES: sink();
}

void constexpr_init_statement_cases() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (constexpr bool HasInt = requires { sizeof(int); }; HasInt) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (requires { sizeof(long); })
      sink();
  }
  // CHECK-FIXES: if constexpr (constexpr bool HasInt = requires { sizeof(int); }; (HasInt) && (requires { sizeof(long); }))
  // CHECK-FIXES: sink();
}

template <typename T> void dependent_requires_outer_is_fixable() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (requires { typename T::type; }) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (true)
      sink();
  }
  // CHECK-FIXES: if constexpr ((requires { typename T::type; }) && (true))
  // CHECK-FIXES: sink();
}

template <bool B, typename T> void dependent_requires_after_bool_is_fixable() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (B) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (requires { typename T::type; })
      sink();
  }
  // CHECK-FIXES: if constexpr ((B) && (requires { typename T::type; }))
  // CHECK-FIXES: sink();
}

void attribute_cases(bool B1, bool B2) {
  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (B1) {
    // CHECK-MESSAGES-NOT: :[[@LINE+1]]:5: warning: nested 'if' statements can be merged together
    [[likely]] if (B2)
      sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  [[likely]] if (B1) {
    if (B2)
      sink();
  }

  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (B1) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (B2)
      sink();
  }
  // CHECK-FIXES: if ((B1) && (B2))
  // CHECK-FIXES: sink();
}
