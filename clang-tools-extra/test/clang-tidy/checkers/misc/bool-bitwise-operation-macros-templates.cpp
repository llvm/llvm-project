// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

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
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a IDENT(& b);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    IDENT(a |=) b;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a IDENT(&= b);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // change operator - GOOD
    IDENT(a) | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: IDENT(a) || b;
    a & IDENT(b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && IDENT(b);
    IDENT(a) & IDENT(b);
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: IDENT(a) && IDENT(b);

    // insert `)` - BAD
    bool c = true, e = false;
    a && b | IDENT(c &&) e;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // insert `)` - GOOD
    a && b | c IDENT(&& e);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && (b || c) IDENT(&& e);

    // insert `(` - BAD
    a IDENT(&& b) | c && e;
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // insert `(` - GOOD
    IDENT(a &&) b | c && e;
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: IDENT(a &&) (b || c) && e;

    bool ab = false;
    // insert ` = a` - BAD
    CAT(a, b) &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // insert ` = a`- GOOD
    b &= CAT(a, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: b = b && CAT(a, b);
}

void bad_in_macro_fixit() {
    bool a = true, b = false;

    // FIXME: implement fixit for all of these cases
    
    a MY_OR b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    a MY_AND b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    a MY_OR_ASSIGN b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    a MY_AND_ASSIGN b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    IDENT(a &= b);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
}

template<typename T>
void good_in_unreachable_template(T a, T b) {
    a | b;
    a & b;
    a |= b;
    a &= b;
}

template<typename T>
int bad_in_template(T a, T b) {
    bool c = false;
    // FIXME: at least warning should be provided in these cases
    // a | b;
    // a & b;
    // a |= b;
    // a &= b;
    // c &= a;
    return 0;
}

template<typename T>
int bad_in_template_lambda_captured(T a, T b) {
    [=] mutable {
        bool c = false;
        // FIXME: at least warning should be provided in these cases
        // a | b;
        // a & b;
        // a |= b;
        // b &= a;
    }();
    return 0;
}

int dummy = bad_in_template(true, false) + bad_in_template_lambda_captured(false, true);
