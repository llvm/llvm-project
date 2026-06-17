// RUN: %check_clang_tidy -std=c++11-or-later %s readability-redundant-nested-if %t -- -- \
// RUN:   -I %S -fno-delayed-template-parsing

#include "Inputs/redundant-nested-if/common.h"

void two_if_chain() {
  // CHECK-NOTES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    // CHECK-NOTES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1))
      sink();
  }
}

void long_if_chain() {
  // CHECK-NOTES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (cond()) {
    // CHECK-NOTES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (cond(1)) {
      // CHECK-NOTES: :[[@LINE+1]]:7: note: nested 'if' statement to merge declared here
      if (cond(2))
        sink();
    }
  }
}
