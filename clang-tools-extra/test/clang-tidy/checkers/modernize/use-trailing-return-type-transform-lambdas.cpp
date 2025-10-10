// RUN: %check_clang_tidy -check-suffix=ALL -std=c++11-or-later %s modernize-use-trailing-return-type %t --\
// RUN:   -config="{CheckOptions: {modernize-use-trailing-return-type.TransformLambdas: all, \
// RUN:                            modernize-use-trailing-return-type.TransformFunctions: false}}" \
// RUN:   -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffix=NOAUTO -std=c++11-or-later %s modernize-use-trailing-return-type %t --\
// RUN:   -config="{CheckOptions: {modernize-use-trailing-return-type.TransformLambdas: all_except_auto, \
// RUN:                            modernize-use-trailing-return-type.TransformFunctions: false}}" \
// RUN:   -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -check-suffix=NONE -std=c++11-or-later %s modernize-use-trailing-return-type %t --\
// RUN:   -config="{CheckOptions: {modernize-use-trailing-return-type.TransformLambdas: none, \
// RUN:                            modernize-use-trailing-return-type.TransformFunctions: true}}" \
// RUN:   -- -fno-delayed-template-parsing

namespace std {
    template <typename T>
    class vector {};

    class string {};
} // namespace std

void test_lambda_positive() {
  auto l01 = [] {};
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:14: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:14: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l01 = [] -> void {};
  auto l1 = []() {};
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l1 = []() -> void {};
  auto l2 = []() { return 42; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l2 = []() -> int { return 42; };
  auto l3 = [](int x, double y) { return x * y; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l3 = [](int x, double y) -> double { return x * y; };

  int capture_int = 10;
  double capture_double = 3.14;
  int* capture_ptr = nullptr;
  
  auto l4 = [capture_int]() { return capture_int; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l4 = [capture_int]() -> int { return capture_int; };
  auto l5 = [capture_int, &capture_double](char c) { return capture_int + capture_double + c; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l5 = [capture_int, &capture_double](char c) -> double { return capture_int + capture_double + c; };
  auto l6 = [capture_int]() constexpr mutable noexcept { return ++capture_int; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l6 = [capture_int]() constexpr mutable noexcept -> int { return ++capture_int; };
  auto l7 = [&capture_ptr]() { return capture_ptr; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l7 = [&capture_ptr]() -> int * { return capture_ptr; };
  auto l8 = [&capture_int]() { return capture_int; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l8 = [&capture_int]() -> int { return capture_int; };
  auto l9 = [] { return std::vector<std::vector<int>>{}; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l9 = [] -> std::vector<std::vector<int>> { return std::vector<std::vector<int>>{}; };
  auto l10 = [] { const char* const * const * const ptr = nullptr; return ptr; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:14: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-NOAUTO: :[[@LINE-2]]:14: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-ALL: auto l10 = [] -> const char *const *const * { const char* const * const * const ptr = nullptr; return ptr; };
}

// In c++11 mode we can not write 'auto' type, see *-cxx14.cpp for fixes.
template <template <typename> class C>
void test_lambda_positive_template() {
  auto l1 = []() { return C<int>{}; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  auto l2 = []() { return 0; };
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
}

void test_lambda_negative() {
  auto l1_good = [](int arg) -> int { return 0; };  
}

// this function is solely used to not to get "wrong config error" from the check.
int f();
// CHECK-MESSAGES-NONE: :[[@LINE-1]]:5: warning: use a trailing return type for this function [modernize-use-trailing-return-type]
// CHECK-FIXES-NONE: auto f() -> int;
