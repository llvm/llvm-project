// RUN: %check_clang_tidy -std=c++98-or-later --extra-arg=-Wno-error=parentheses %s bugprone-chained-comparison %t -- -config="{CheckOptions: {bugprone-chained-comparison.IgnoreMacros: true}}"

#define CHAINED_COMPARE(a, b, c) (a < b < c)

void macro_test(int x, int y, int z) {
    bool result = CHAINED_COMPARE(x, y, z);
}

void normal_test(int x, int y, int z) {
    bool result = x < y < z;
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: chained comparison 'v0 < v1 < v2' may generate unintended results
}
