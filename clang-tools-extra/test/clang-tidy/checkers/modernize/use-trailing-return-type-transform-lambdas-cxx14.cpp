// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-use-trailing-return-type %t -- -- -fno-delayed-template-parsing

namespace std {
    template <typename T>
    class vector {};

    class string {};
} // namespace std

void test_lambda_positive() {
  auto l1 = [](auto x) { return x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l1 = [](auto x) -> auto { return x; };
}

template <template <typename> class C>
void test_lambda_positive_template() {
  auto l1 = []() { return C<int>{}; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l1 = []() -> auto { return C<int>{}; };
  auto l2 = []() { return 0; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l2 = []() -> auto { return 0; };
}
