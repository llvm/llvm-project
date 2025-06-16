// RUN: %check_clang_tidy %s performance-bool-bitwise-operation %t \
// RUN:   -config="{CheckOptions: { \
// RUN:     performance-bool-bitwise-operation.StrictMode: false }}"

bool function_with_possible_side_effects();

void bad_possible_side_effects() {
    bool a = true, b = false;

    a | function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean values instead of bitwise operator '|' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a || function_with_possible_side_effects();

    a & function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean values instead of bitwise operator '&' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a && function_with_possible_side_effects();

    a |= function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean variable 'a' instead of bitwise operator '|=' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a || function_with_possible_side_effects();

    a &= function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean variable 'a' instead of bitwise operator '&=' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a && function_with_possible_side_effects();

    bool c = true;

    a &= function_with_possible_side_effects() && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean variable 'a' instead of bitwise operator '&=' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a && function_with_possible_side_effects() && c;

    a &= b && function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean variable 'a' instead of bitwise operator '&=' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b && function_with_possible_side_effects();

    a |= function_with_possible_side_effects() || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean variable 'a' instead of bitwise operator '|=' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a || function_with_possible_side_effects() || c;

    a |= b || function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean variable 'a' instead of bitwise operator '|=' [performance-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b || function_with_possible_side_effects();
}

void bad_definitely_side_effects() {
    bool a = true, b = false;
    int acc = 0;

    a | (acc++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean values instead of bitwise operator '|' [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a & (acc++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean values instead of bitwise operator '&' [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= (acc++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean variable 'a' instead of bitwise operator '|=' [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a &= (acc++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean variable 'a' instead of bitwise operator '&=' [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    bool c = true;

    a &= (acc++, b) && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean variable 'a' instead of bitwise operator '&=' [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a &= b && (acc++, c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean variable 'a' instead of bitwise operator '&=' [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= (acc++, b) || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean variable 'a' instead of bitwise operator '|=' [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= b || (acc++, c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean variable 'a' instead of bitwise operator '|=' [performance-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

