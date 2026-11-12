// RUN: %check_clang_tidy -check-suffixes=ALLOWBOOL -std=c++98-or-later %s \
// RUN:   readability-redundant-nested-if %t -- \
// RUN:   -config='{CheckOptions: {readability-redundant-nested-if.AllowUserDefinedBoolConversion: true}}' -- \
// RUN:   -I %S -fno-delayed-template-parsing

#include "Inputs/redundant-nested-if/common.h"

void user_defined_bool_conversion_cases() {
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:3: warning: nested 'if' statements can be merged together
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:5: note: nested 'if' statement to merge declared here
  if (make_bool_like()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-ALLOWBOOL: if ((static_cast<bool>(make_bool_like())) && (cond(1)))
  // CHECK-FIXES-ALLOWBOOL: sink();

  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:3: warning: nested 'if' statements can be merged together
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:5: note: nested 'if' statement to merge declared here
  if (cond(1)) {
    if (make_bool_like())
      sink();
  }
  // CHECK-FIXES-ALLOWBOOL: if ((cond(1)) && (static_cast<bool>(make_bool_like())))
  // CHECK-FIXES-ALLOWBOOL: sink();
}
