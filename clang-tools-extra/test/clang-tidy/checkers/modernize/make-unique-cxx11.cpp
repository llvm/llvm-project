// RUN: %check_clang_tidy -std=c++11 %s modernize-make-unique %t

#include <memory>
// CHECK-FIXES: #include <memory>

void f() {
  auto my_ptr = std::unique_ptr<int>(new int(1));
  // CHECK-FIXES: auto my_ptr = std::unique_ptr<int>(new int(1));
}
