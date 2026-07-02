// RUN: %check_clang_tidy -std=c++26-or-later %s readability-redundant-nested-if %t -- -- \
// RUN:   -I %S -fno-delayed-template-parsing

#include "Inputs/redundant-nested-if/common.h"

template <typename T> void constexpr_template_static_assert() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (sizeof(T) == 1) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (false) {
      static_assert(false, "discarded in template context");
    }
  }
  // CHECK-FIXES: if constexpr ((sizeof(T) == 1) && (false))
  // CHECK-FIXES: static_assert(false, "discarded in template context");
}
