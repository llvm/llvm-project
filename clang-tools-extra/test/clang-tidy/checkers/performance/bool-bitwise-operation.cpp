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

void bad_with_priors_nontraditional() {
    bool a = false, b = true, c = true;
    a and b bitor c;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a and (b or c);
    a and b bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a and b and c;
    a or b bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a or b and c;
    a or b bitor c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a or b or c;
    b bitor c and a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: (b or c) and a;
}

void bad_with_priors2_nontraditional() {
    bool a = false, b = true, c = true;
    a xor b bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a xor (b and c);
    a bitor b bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a or b and c;
    b bitand c xor a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: (b and c) xor a;
}

void bad_with_priors_compound_nontraditional() {
    bool a = false, b = true, c = true;
    a and_eq b or c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a and (b or c);
    a or_eq b or c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a or b or c;
    a and_eq b and c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a and b and c;
    a or_eq b and c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a or b and c;
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

// TODO: nontraditional
#define MY_OR |
#define MY_AND &
#define MY_OR_ASSIGN |=
#define MY_AND_ASSIGN &=

#define MY_OR_FUNC(a, b) ((a) | (b))
#define MY_AND_FUNC(a, b) ((a) & (b))
#define MY_OR_ASSIGN_FUNC(a, b) ((a) |= (b))
#define MY_AND_ASSIGN_FUNC(a, b) ((a) &= (b))

#define IDENT(a) (a)

// TODO: check that braces will not be settled inside the macro
// TODO: check that =a will not be sellted inside the macro(for both cases)

void bad_in_macro() {
    bool a = true, b = false;

    // TODO: same for braces as in math check

    a MY_OR b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a || b;
    a MY_AND b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a && b;
    a MY_OR_ASSIGN b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b;
    a MY_AND_ASSIGN b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b;

    // TODO: same but for partial(with hints and with no hints)
    MY_OR_FUNC(a, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    MY_AND_FUNC(a, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    MY_OR_ASSIGN_FUNC(a, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    MY_AND_ASSIGN_FUNC(a, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    IDENT(a | b);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: IDENT(a || b);
    IDENT(a & b);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-FIXES: IDENT(a && b);
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


