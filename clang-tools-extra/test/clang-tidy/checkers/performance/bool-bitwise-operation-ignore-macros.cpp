// RUN: %check_clang_tidy %s performance-bool-bitwise-operation %t \
// RUN:   -config="{CheckOptions: { \
// RUN:     performance-bool-bitwise-operation.IgnoreMacros: true }}"

#define MY_OR |
#define MY_AND &
#define MY_OR_ASSIGN |=
#define MY_AND_ASSIGN &=
#define MY_LOG_AND &&

#define CAT(a, b) a ## b
#define IDENT(a) a

void bad_in_macro() {
    bool a = true, b = false;

    // change operator - BAD
    IDENT(a |) b;
    a IDENT(& b);
    IDENT(a |=) b;
    a IDENT(&= b);

    // change operator - GOOD
    IDENT(a) | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean values instead of bitwise operator '|' [performance-bool-bitwise-operation]
    // CHECK-FIXES: IDENT(a) || b;
    a & IDENT(b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean values instead of bitwise operator '&' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a && IDENT(b);
    IDENT(a) & IDENT(b);
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean values instead of bitwise operator '&' [performance-bool-bitwise-operation]
    // CHECK-FIXES: IDENT(a) && IDENT(b);

    // insert `)` - BAD
    bool c = true, e = false;
    a && b | IDENT(c &&) e;

    // insert `)` - GOOD
    a && b | c IDENT(&& e);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '||' for boolean values instead of bitwise operator '|' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a && (b || c) IDENT(&& e);

    // insert `(` - BAD
    a IDENT(&& b) | c && e;

    // insert `(` - GOOD
    IDENT(a &&) b | c && e;
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use logical operator '||' for boolean values instead of bitwise operator '|' [performance-bool-bitwise-operation]
    // CHECK-FIXES: IDENT(a &&) (b || c) && e;

    bool ab = false;
    // insert ` = a` - BAD
    CAT(a, b) &= b;

    // insert ` = a`- GOOD
    b &= CAT(a, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean variable 'b' instead of bitwise operator '&=' [performance-bool-bitwise-operation]
    // CHECK-FIXES: b = b && CAT(a, b);
}

