// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

struct A {
    int first;
    bool second;
};

void normal() {
    A a {100, false};
    int b = 200;

    a.first | b;
    a.first & b;
    a.first |= b;
    a.first &= b;
}

void bad() {
    A a {-1, true};
    bool b = false;

    a.second | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a.second || b;
    a.second & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a.second && b;
    a.second |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a.second = a.second || b;
    a.second &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a.second = a.second && b;
}

void bad_two_lines() {
    A a {-1, true};
    bool b = false;

    a.
      second | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: second || b;
    a.
    second & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: second && b;
    a.
     second |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: second = a.second || b;
    a
    .
    second &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: second = a.second && b;
    a
    .second &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: .second = a.second && b;
}

void bad_side_effects_volatile() {
    volatile A a {-1, true};
    bool b = false;

    a.second | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a.second & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a.second |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a.second &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

struct VolatileA {
    volatile bool second;
};

void bad_side_effects_volatile2() {
    VolatileA a {true};
    bool b = false;

    a.second | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a.second & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a.second |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a.second &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void bad_arrow() {
    A a {-1, true};
    auto* pa = &a;
    bool b = false;

    pa->second | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: pa->second || b;
    pa->second & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: pa->second && b;
    pa->second |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: pa->second = pa->second || b;
    pa->second &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: pa->second = pa->second && b;
}

struct B {
    bool& access();
};

void bad_no_fixit() {
    B b;
    bool c = false;
    b.access() |= c;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    b.access() &= c;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    auto* pb = &b;
    pb->access() |= c;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    pb->access() &= c;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

struct A_with_ptrs {
    int* first;
    bool* second;
};

void bad_pointers() {
    int first = -1;
    bool second = true;
    A_with_ptrs a {&first, &second};
    bool b = false;

    *a.second |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *a.second = *a.second || b;
    *a.second &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *a.second = *a.second && b;
}

struct BoolWrapper {
    bool value;
    bool& operator*() { return value; }
};

struct A_with_wrapper {
    int first;
    BoolWrapper second;
};

void bad_user_defined_deref() {
    int first = -1;
    BoolWrapper second {true};
    A_with_wrapper a {first, second};
    bool b = false;

    *a.second |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    *a.second &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}
