// RUN: %check_clang_tidy -std=c++11  %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++14 %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++17  %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++20  %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffix=23 %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffix=23-STATIC %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR', modernize-use-constexpr.StaticConstexprString: 'STATIC_CXPR'}}"

static int f1() { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: function 'f1' can be declared 'constexpr' [modernize-use-constexpr]
// CHECK-FIXES: static CXPR int f1() { return 0; }
// CHECK-MESSAGES-23: :[[@LINE-3]]:12: warning: function 'f1' can be declared 'constexpr' [modernize-use-constexpr]
// CHECK-FIXES-23: static CXPR int f1() { return 0; }
// CHECK-MESSAGES-23-STATIC: :[[@LINE-5]]:12: warning: function 'f1' can be declared 'constexpr' [modernize-use-constexpr]
// CHECK-FIXES-23-STATIC: static CXPR int f1() { return 0; }

#define FUNC(N) void func##N()
FUNC(0) {
    static int f1 = 1;
    static const int f2 = 2;
    // CHECK-MESSAGES-23: :[[@LINE-1]]:22: warning: variable 'f2' can be declared 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES-23: static CXPR int f2 = 2;
    // CHECK-MESSAGES-23-STATIC: :[[@LINE-3]]:22: warning: variable 'f2' can be declared 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES-23-STATIC: static CXPR int f2 = 2;
    const int f3 = 3;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: variable 'f3' can be declared 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES: CXPR int f3 = 3;
    // CHECK-MESSAGES-23: :[[@LINE-3]]:15: warning: variable 'f3' can be declared 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES-23: static CXPR  int f3 = 3;
    // CHECK-MESSAGES-23-STATIC: :[[@LINE-5]]:15: warning: variable 'f3' can be declared 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES-23-STATIC: STATIC_CXPR  int f3 = 3;
}

