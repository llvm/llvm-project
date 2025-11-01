// RUN: %check_clang_tidy -std=c++11 %s modernize-use-constexpr %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++14 %s modernize-use-constexpr %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++17 %s modernize-use-constexpr %t -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++20 %s modernize-use-constexpr %t -- -- -fno-delayed-template-parsing


#define FUNC(N) void func##N()
FUNC(0) {
    static int f1 = 1;
    static const int f2 = 2;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: declare variable 'f2' as 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES: static constexpr int f2 = 2;
    const int f3 = 3;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: declare variable 'f3' as 'constexpr' [modernize-use-constexpr]
    // CHECK-FIXES: constexpr int f3 = 3;
}

