// RUN: %check_clang_tidy %s performance-bool-bitwise-operation %t

bool& normal() {
    int a = 100, b = 200;

    a | b;
    a & b;
    a |= b;
    a &= b;

    a bitor b;
    a bitand b;
    a or_eq b;
    a and_eq b;

    static bool st = false;
    return st;
}

void bad() {
    bool a = true, b = false;
    a | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a || b;
    a & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a && b;
    a |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b;
    a &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b;
}

void bad_volatile_bool() {
    bool a = true;
    volatile bool b = false;
    a | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void bad_nontraditional() {
    bool a = true, b = false;
    a bitor b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a or b;
    a bitand b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a and b;
    a or_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a or b;
    a and_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a and b;
}

void bad_with_priors() {
    bool a = false, b = true, c = true;
    a && b | c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a && (b || c);
    a && b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a && b && c;
    a || b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a || b && c;
    a || b | c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a || b || c;
    b | c && a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: (b || c) && a;
}

void bad_with_priors2() {
    bool a = false, b = true, c = true;
    a ^ b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a ^ (b && c);
    a | b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a || b && c;
    b & c ^ a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: (b && c) ^ a;
}

void bad_with_priors_compound() {
    bool a = false, b = true, c = true;
    a &= b || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a && (b || c);
    a |= b || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b || c;
    a &= b && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b && c;
    a |= b && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b && c;
}

void bad_with_priors_compound2() {
    // TODO: ^ and |
}

void bad_no_fixit() {
    bool b = false;
    normal() |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    normal() &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

// TODO: all the same tests as in math-parentheses check
void bad_in_macro___() {
#if 0
    const std::string Input = R"cc(
#define M(a,b) (true & a) * (true | b)
    int f() { return M(false, false); }
    )cc";
#endif
// TODO: implement this(make sure no fix were provided)
#if 0
    const std::string Input = R"cc(
#define M(a,b) (true a) * (true b)
    int f() { return M(& false, | false); }
    )cc";
    const std::string Expected = R"cc(
#define M(a,b) (true a) * (true b)
    int f() { return M(&& false, || false); }
    )cc";
#endif
// TODO: implement this
}

template<typename T>
void good_in_unreachable_template(T a, T b) {
    a | b;
    a & b;
    a |= b;
    a &= b;
}

// TODO: test for in template
// TODO: test for type in typedef
// TODO: test for expressions in parentheses


