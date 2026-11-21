// RUN: %check_clang_tidy -check-suffix=BEFORE-CXX23 -std=c++11,c++14,c++17 %s modernize-use-trailing-return-type %t -- -- -Wno-c++23-extensions
// RUN: %check_clang_tidy -check-suffixes=BEFORE-CXX23,CXX20 -std=c++20 %s modernize-use-trailing-return-type %t -- -- -Wno-c++23-extensions
// RUN: %check_clang_tidy -check-suffixes=CXX23 -std=c++23-or-later %s modernize-use-trailing-return-type %t

// A lambda can only omit its parameter clause before a trailing return type
// since C++23 (P1102R2), so the fix has to add '()' in earlier language modes.

#define LAMBDA_BODY { return 1; }

void test_no_parameter_clause() {
  int x = 0;

  auto l1 = [] {};
  // CHECK-MESSAGES-BEFORE-CXX23: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-CXX23: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-BEFORE-CXX23: auto l1 = []() -> void {};
  // CHECK-FIXES-CXX23: auto l1 = [] -> void {};

  auto l2 = [] { return 1; };
  // CHECK-MESSAGES-BEFORE-CXX23: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-CXX23: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-BEFORE-CXX23: auto l2 = []() -> int { return 1; };
  // CHECK-FIXES-CXX23: auto l2 = [] -> int { return 1; };

  auto l3 = [&x] /* comment */ { return x; };
  // CHECK-MESSAGES-BEFORE-CXX23: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-CXX23: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-BEFORE-CXX23: auto l3 = [&x]() -> int /* comment */ { return x; };
  // CHECK-FIXES-CXX23: auto l3 = [&x] -> int /* comment */ { return x; };

  // Specifiers would have to follow '()', so don't offer a fix before C++23.
  auto l4 = [x] mutable { return x; };
  // CHECK-MESSAGES-BEFORE-CXX23: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-CXX23: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-BEFORE-CXX23: auto l4 = [x] mutable { return x; };
  // CHECK-FIXES-CXX23: auto l4 = [x] mutable -> int { return x; };

#if __cplusplus < 202302L
  auto l6 = []{ return 1; };
  // CHECK-MESSAGES-BEFORE-CXX23: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-BEFORE-CXX23: auto l6 = []() -> int{ return 1; };

  auto l7 = [] LAMBDA_BODY;
  // CHECK-MESSAGES-BEFORE-CXX23: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-BEFORE-CXX23: auto l7 = [] LAMBDA_BODY;
#endif

#if __cplusplus >= 202002L
  auto l5 = []<typename T = int> { return T{}; };
  // CHECK-MESSAGES-CXX20: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-CXX23: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-CXX20: auto l5 = []<typename T = int>() -> auto { return T{}; };
  // CHECK-FIXES-CXX23: auto l5 = []<typename T = int> -> auto { return T{}; };

  // A requires clause would have to come before '()', so don't offer a fix
  // before C++23.
  auto l8 = []<typename T = int> requires true { return T{}; };
  // CHECK-MESSAGES-CXX20: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-MESSAGES-CXX23: :[[@LINE-2]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES-CXX20: auto l8 = []<typename T = int> requires true { return T{}; };
#endif
}
