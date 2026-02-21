// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t \
// RUN:   -config="{CheckOptions: { \
// RUN:     misc-bool-bitwise-operation.UnsafeMode: true }}"

bool function_with_possible_side_effects();

void bad_possible_side_effects() {
    bool a = true, b = false;

    a | function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || function_with_possible_side_effects();

    a & function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && function_with_possible_side_effects();

    a |= function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || function_with_possible_side_effects();

    a &= function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && function_with_possible_side_effects();

    bool c = true;

    a &= function_with_possible_side_effects() && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && function_with_possible_side_effects() && c;

    a &= b && function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b && function_with_possible_side_effects();

    a |= function_with_possible_side_effects() || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || function_with_possible_side_effects() || c;

    a |= b || function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b || function_with_possible_side_effects();
}

void bad_definitely_side_effects() {
    bool a = true, b = false;
    int acc = 0;

    a | (acc++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a & (acc++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= (acc++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a &= (acc++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    bool c = true;

    a &= (acc++, b) && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a &= b && (acc++, c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= (acc++, b) || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= b || (acc++, c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void bad_definitely_side_effects_notes() {
    bool a = true, b = false;
    int acc1 = 0, acc2 = 0;

    a | (acc2++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:9: note: extract the right operand to a variable

    b & (acc2++, a);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:9: note: extract the right operand to a variable

    a |= (acc2++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:10: note: extract the right operand to a variable

    a &= (acc2++, b);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{.*}}: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:10: note: extract the right operand to a variable
}
