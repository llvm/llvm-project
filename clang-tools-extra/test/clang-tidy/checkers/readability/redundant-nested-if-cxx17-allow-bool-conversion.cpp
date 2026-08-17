// RUN: %check_clang_tidy -check-suffixes=ALLOWBOOL -std=c++17-or-later %s \
// RUN:   readability-redundant-nested-if %t -- \
// RUN:   -config='{CheckOptions: {readability-redundant-nested-if.AllowUserDefinedBoolConversion: true}}' -- \
// RUN:   -I %S -fno-delayed-template-parsing

#include "Inputs/redundant-nested-if/common.h"

void declaration_condition_boollike_cases() {
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:3: warning: nested 'if' statements can be merged together
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:5: note: nested 'if' statement to merge declared here
  if (auto Guard = make_bool_like()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-ALLOWBOOL: if (auto Guard = make_bool_like(); static_cast<bool>(Guard) && (cond(1)))
  // CHECK-FIXES-ALLOWBOOL: sink();

  if (bool X = COND_MACRO) {
    if (cond(1))
      sink();
  }
}
