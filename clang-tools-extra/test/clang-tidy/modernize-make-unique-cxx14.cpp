// RUN: %check_clang_tidy %s modernize-make-unique %t -- -- -std=c++14 \
// RUN:   -I%S/Inputs/modernize-smart-ptr

#include "unique_ptr.h"
// CHECK-FIXES: #include <memory>

void f() {
  auto my_ptr = std::unique_ptr<int>(new int(1));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use std::make_unique instead
  // CHECK-FIXES: auto my_ptr = std::make_unique<int>(1);
}
