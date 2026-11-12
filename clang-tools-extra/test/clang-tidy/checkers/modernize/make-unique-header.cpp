// RUN: %check_clang_tidy %s modernize-make-unique %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:     {modernize-make-unique.MakeSmartPtrFunction: 'my::MakeUnique', \
// RUN:      modernize-make-unique.MakeSmartPtrFunctionHeader: 'make_unique_util.h' \
// RUN:     }}"

#include <memory>
// CHECK-FIXES: #include "make_unique_util.h"

void f() {
  std::unique_ptr<int> P1 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use my::MakeUnique instead
  // CHECK-FIXES: std::unique_ptr<int> P1 = my::MakeUnique<int>();
}
