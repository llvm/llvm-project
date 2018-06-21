// RUN: %check_clang_tidy %s modernize-make-unique %t -- -- -std=c++11 \
// RUN:   -I%S/Inputs/modernize-smart-ptr

#include "unique_ptr.h"
// CHECK-FIXES: #include "unique_ptr.h"

void f() {
  auto my_ptr = std::unique_ptr<int>(new int(1));
  // CHECK-FIXES: auto my_ptr = std::unique_ptr<int>(new int(1));
}
