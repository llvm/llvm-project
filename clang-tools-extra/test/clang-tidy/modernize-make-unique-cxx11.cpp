// RUN: %check_clang_tidy -std=c++11 %s modernize-make-unique %t -- -- -I %S/Inputs/modernize-smart-ptr

#include "unique_ptr.h"
// CHECK-FIXES: #include "unique_ptr.h"

void f() {
  auto my_ptr = std::unique_ptr<int>(new int(1));
  // CHECK-FIXES: auto my_ptr = std::unique_ptr<int>(new int(1));
}
