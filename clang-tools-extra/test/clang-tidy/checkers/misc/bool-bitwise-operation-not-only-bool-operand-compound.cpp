// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

void general(unsigned flags, bool value) {
    flags |= value;

    unsigned mask = 0b1100;
    bool result = flags &= mask;
    auto result2 = flags &= mask;
    result = flags |= flags << 1;
}

void assign_to_boolean(unsigned flags, bool value) {
    value |= flags << 1;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: value = value || flags << 1;
    value |= (flags << 1) | (flags << 2);
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: value = value || (flags << 1) || (flags << 2);
}
