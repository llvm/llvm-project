// RUN: %check_clang_tidy %s modernize-make-unique %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:     [{key: modernize-make-unique.MakeSmartPtrFunction, \
// RUN:       value: 'my::MakeUnique'}, \
// RUN:      {key: modernize-make-unique.MakeSmartPtrFunctionHeader, \
// RUN:       value: 'make_unique_util.h'} \
// RUN:     ]}" \
// RUN:   -- -I %S/Inputs/modernize-smart-ptr

#include "unique_ptr.h"
// CHECK-FIXES: #include "make_unique_util.h"

void f() {
  std::unique_ptr<int> P1 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use my::MakeUnique instead
  // CHECK-FIXES: std::unique_ptr<int> P1 = my::MakeUnique<int>();
}
