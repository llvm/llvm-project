// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t -check-suffixes=,ENABLED \
// RUN:   -config="{CheckOptions: { \
// RUN:     misc-bool-bitwise-operation.ParenCompounds: true }}"
// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t -check-suffixes=,DISABLED \
// RUN:   -config="{CheckOptions: { \
// RUN:     misc-bool-bitwise-operation.ParenCompounds: false }}"

// Test with ParenCompounds=true and false: braces should be added around RHS for compound operators when enabled

void test_brace_rhs_enabled() {
    bool a = true, b = false, c = true;

    // Compound operators should have braces around RHS
    a &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b;

    a |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b;

    // Non-compound operators should not have braces around RHS
    a & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && b;

    a | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || b;

    // Compound operators with more complex RHS
    a &= b && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b && c;

    a |= b || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b || c;

    // Compound operators with bitwise operators in RHS
    a &= b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b && c;

    a |= b | c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b || c;

    // Case where ParensExpr is the RHS (&= with || RHS) - should not add double parentheses
    a &= b || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && (b || c);

    // Case where ParensExpr is the RHS (|= with && RHS) - should not add double parentheses
    a |= b && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES-ENABLED: a = a || (b && c);
    // CHECK-FIXES-DISABLED: a = a || b && c;

    // Case where ParensExpr is NOT the RHS (&= with | RHS) - braces should be added by ParenCompounds when enabled
    a &= b | c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES-ENABLED: a = a && (b || c);
    // CHECK-FIXES-DISABLED: a = a && b || c;

    // Case where ParensExpr is NOT the RHS (|= with & RHS) - braces should be added by ParenCompounds when enabled
    a |= b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES-ENABLED: a = a || (b && c);
    // CHECK-FIXES-DISABLED: a = a || b && c;

    // Case where ParensExpr is NOT the RHS (&= with == RHS) - braces should be added by ParenCompounds when enabled
    a &= b == c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES-ENABLED: a = a && (b == c);
    // CHECK-FIXES-DISABLED: a = a && b == c;

    // Case where ParensExpr is NOT the RHS (|= with != RHS) - braces should be added by ParenCompounds when enabled
    a |= b != c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES-ENABLED: a = a || (b != c);
    // CHECK-FIXES-DISABLED: a = a || b != c;

    a &= (b || c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && (b || c);

    a |= (b && c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || (b && c);
}
