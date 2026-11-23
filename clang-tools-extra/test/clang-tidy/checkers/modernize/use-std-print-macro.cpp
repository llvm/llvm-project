// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-std-print %t

#include <cstdio>
// CHECK-FIXES: #include <print>

#define WRAP_MSG(msg) msg

void macro_argument_include(int n) {
  WRAP_MSG(printf("value %d", n));
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: WRAP_MSG(std::print("value {}", n));
}
