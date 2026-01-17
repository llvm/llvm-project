// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t \
// RUN:   -config="{CheckOptions: { \
// RUN:     misc-bool-bitwise-operation.StrictMode: false }}"

// Test with StrictMode=false: warnings should NOT be shown when fixits don't exist

bool function_with_possible_side_effects();

void test_strict_mode_disabled() {
    bool a = true, b = false;

    // Case 1: Side effects in RHS - no fixit, no warning should be shown
    a | function_with_possible_side_effects();
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:{{.*}}: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]

    a & function_with_possible_side_effects();
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:{{.*}}: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]

    a |= function_with_possible_side_effects();
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:{{.*}}: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]

    a &= function_with_possible_side_effects();
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:{{.*}}: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]

    // Case 2: Volatile operands - no fixit, no warning should be shown
    volatile bool v = false;
    a | v;
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:{{.*}}: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]

    a & v;
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:{{.*}}: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]

    a |= v;
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:{{.*}}: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]

    a &= v;
    // CHECK-MESSAGES-NOT: :[[@LINE-1]]:{{.*}}: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]

    // Case 3: Normal case with fixit - warning should still be shown with fixit
    a | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || b;

    a & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && b;
}
