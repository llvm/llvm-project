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
#define MY_LOG_AND &&

#define MY_OR_FUNC(a, b) ((a) | (b))
#define MY_AND_FUNC(a, b) ((a) & (b))
#define MY_OR_ASSIGN_FUNC(a, b) ((a) |= (b))
#define MY_AND_ASSIGN_FUNC(a, b) ((a) &= (b))

#define IDENT(a) (a)

bool global_flag = false;
int sink(int);
#define FUN(ARG) (sink(ARG))
#define FUN2(ARG) sink((ARG))
#define FUN3(ARG) sink(ARG)
#define FUN4(ARG) sink(true | ARG)
#define FUN5(ARG) sink(false && ARG)
#define FUN6(ARG) sink(global_flag |= ARG)

#define M1(a,b) (true & a) && (true | b)
#define M2(a,b) (true a) && (true b)

// FIXME: implement fixit hint for simple macro cases
void bad_in_macro() {
    bool a = true, b = false;

    a MY_OR b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a MY_AND b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a MY_OR_ASSIGN b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a MY_AND_ASSIGN b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // TODO: bool q = true MY_AND false MY_OR true MY_AND false MY_OR true MY_AND false MY_AND true MY_OR false;

    bool q = (true MY_LOG_AND false MY_OR true) MY_LOG_AND (false MY_OR true MY_LOG_AND (false MY_LOG_AND true MY_OR false));
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:67: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:112: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-4]]:{{.*}}: note: FIX-IT applied suggested code changes

    bool r = FUN(true | false && true);
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    bool s = FUN2(true | false && true);
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    bool t = FUN3(true | false && true);
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    bool u = FUN4(true && false);
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    bool v = FUN5(false | true);
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    bool vv = FUN6(true && false);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

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
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    IDENT(a & b);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    bool abc = false;
    IDENT(abc |= b);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    IDENT(abc &= b);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    M1(false, false);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    M2(& false, | false);
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: use logical operator instead of bitwise one for bool [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-3]]:{{.*}}: note: FIX-IT applied suggested code changes
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


