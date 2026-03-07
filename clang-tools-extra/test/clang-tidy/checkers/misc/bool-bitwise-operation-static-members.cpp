// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

struct A {
    static int first;
    static bool second;
};

int A::first = 100;
bool A::second = false;

void normal() {
    int b = 200;

    A::first | b;
    A::first & b;
    A::first |= b;
    A::first &= b;
}

void bad() {
    bool b = false;

    A::second | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: A::second || b;
    A::second & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: A::second && b;
    A::second |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: A::second = A::second || b;
    A::second &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: A::second = A::second && b;
}
