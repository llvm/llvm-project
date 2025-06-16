// RUN: %check_clang_tidy %s performance-bool-bitwise-operation %t

void normal() {
    int a = 100, b = 200;

    a | b;
    a & b;
    a |= b;
    a &= b;
}

void bad() {
    _Bool a = 1, b = 0;
    a | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean values instead of bitwise operator '|' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a || b;
    a & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean values instead of bitwise operator '&' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a && b;
    a |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean variable 'a' instead of bitwise operator '|=' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b;
    a &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean variable 'a' instead of bitwise operator '&=' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b;
}

