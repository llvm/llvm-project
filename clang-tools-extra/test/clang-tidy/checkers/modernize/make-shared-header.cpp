// RUN: %check_clang_tidy %s modernize-make-shared %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:     [{key: modernize-make-shared.MakeSmartPtrFunction, \
// RUN:       value: 'my::MakeShared'}, \
// RUN:      {key: modernize-make-shared.MakeSmartPtrFunctionHeader, \
// RUN:       value: 'make_shared_util.h'} \
// RUN:     ]}" \
// RUN:   -- -I %S/Inputs/smart-ptr

#include "shared_ptr.h"
// CHECK-FIXES: #include "make_shared_util.h"

void f() {
  std::shared_ptr<int> P1 = std::shared_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use my::MakeShared instead
  // CHECK-FIXES: std::shared_ptr<int> P1 = my::MakeShared<int>();
}
