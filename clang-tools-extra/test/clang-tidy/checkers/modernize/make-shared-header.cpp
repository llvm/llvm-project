// RUN: %check_clang_tidy %s modernize-make-shared %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:     {modernize-make-shared.MakeSmartPtrFunction: 'my::MakeShared', \
// RUN:      modernize-make-shared.MakeSmartPtrFunctionHeader: 'make_shared_util.h' \
// RUN:     }}" \
// RUN:   -- -I %S/Inputs/smart-ptr

#include "shared_ptr.h"
// CHECK-FIXES: #include "make_shared_util.h"

void f() {
  std::shared_ptr<int> P1 = std::shared_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use my::MakeShared instead
  // CHECK-FIXES: std::shared_ptr<int> P1 = my::MakeShared<int>();
}
