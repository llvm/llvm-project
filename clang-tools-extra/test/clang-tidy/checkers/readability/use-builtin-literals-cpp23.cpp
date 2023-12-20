// RUN: %check_clang_tidy -std=c++23-or-later %s readability-use-builtin-literals %t

using size_t = decltype(sizeof(void*));
namespace std {
  using size_t = size_t;
}

void warn_and_fix() {

  (size_t)6;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 6uz;
  (std::size_t)6;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 6uz;
  (size_t)6zu;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 6uz;
}
