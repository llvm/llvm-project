// RUN: %check_clang_tidy -std=c++20-or-later %s cppcoreguidelines-missing-std-forward %t -- -- -fno-delayed-template-parsing

#include <utility>

void does_not_forward_auto(auto &&t) {
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: forwarding reference parameter 't' is never forwarded inside the function body
  (void)t;
}

void does_forward_auto(auto &&t) {
  (void)std::forward<decltype(t)>(t);
}

void const_auto_rvalue_reference(const auto &&t) {
  (void)t;
}

template <typename T>
void mixed_parameters(T &&t, auto &&u) {
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: forwarding reference parameter 't' is never forwarded inside the function body
  // CHECK-MESSAGES: :[[@LINE-2]]:37: warning: forwarding reference parameter 'u' is never forwarded inside the function body
  (void)t;
  (void)u;
}
