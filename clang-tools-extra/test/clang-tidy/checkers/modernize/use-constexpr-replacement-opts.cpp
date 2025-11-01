// RUN: %check_clang_tidy -std=c++11  %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++14 %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++17  %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++20  %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR'}}"
// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-constexpr %t -- -config="{CheckOptions: {modernize-use-constexpr.ConstexprString: 'CXPR', modernize-use-constexpr.StaticConstexprString: 'STATIC_CXPR'}}"

static int f1() { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: declare function 'f1' as 'constexpr' [modernize-use-constexpr]
// CHECK-FIXES: static CXPR int f1() { return 0; }

#define FUNC(N) void func##N()
FUNC(0) {
    static int f1 = 1;
    static const int f2 = 2;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: declare variable 'f2' as 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES: static CXPR int f2 = 2;
    const int f3 = 3;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: declare variable 'f3' as 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES: CXPR int f3 = 3;
}

