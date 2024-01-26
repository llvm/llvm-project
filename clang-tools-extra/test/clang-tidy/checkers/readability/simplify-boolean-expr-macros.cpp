// RUN: %check_clang_tidy -check-suffixes=,MACROS %s readability-simplify-boolean-expr %t

// Ignore expressions in macros.
// RUN: %check_clang_tidy %s readability-simplify-boolean-expr %t \
// RUN:     -- -config="{CheckOptions: {readability-simplify-boolean-expr.IgnoreMacros: true}}" \
// RUN:     --

#define NEGATE(expr) !(expr)

bool without_macro(bool a, bool b) {
    return !(!a && b);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: boolean expression can be simplified by DeMorgan's theorem
    // CHECK-FIXES: return a || !b;
}

bool macro(bool a, bool b) {
    return NEGATE(!a && b);
    // CHECK-MESSAGES-MACROS: :[[@LINE-1]]:12: warning: boolean expression can be simplified by DeMorgan's theorem
    // CHECK-FIXES: return NEGATE(!a && b);
}
