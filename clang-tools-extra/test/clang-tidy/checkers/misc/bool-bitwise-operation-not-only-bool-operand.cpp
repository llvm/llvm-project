// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

// The case is taken from the real code in clang/lib/APINotes/APINotesWriter.cpp

void take(bool value) {}
void take_int(int value) {}

void general(unsigned flags, bool value) {
    (flags << 1) | value;
    flags = (flags << 1) | value;
    flags = (flags << 1) | (flags << 2) | value;
    flags = (flags << 1) | (flags << 2) | (flags << 4) | value;
    take_int((flags << 1) | value);
    take_int((flags << 1) | (flags << 2) | value);
}

// FIXME: implement `template<bool bb=true|1>` cases

void assign_to_boolean(unsigned flags, bool value) {
    struct A { bool a = true | 1; };
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct A { bool a = true || 1; };
    struct B { union { bool a = true | 1; }; };
    // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct B { union { bool a = true || 1; }; };
    bool result = (flags << 1) | value;
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool result = (flags << 1) || value;
    bool a = (flags << 2) | value,
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool a = (flags << 2) || value,
         b = (flags << 4) | value,
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: b = (flags << 4) || value,
         c = (flags << 8) | value;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: c = (flags << 8) || value;
    result = (flags << 1) | value;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = (flags << 1) || value;
    take((flags << 1) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: take((flags << 1) || value);
    result = (flags << 1) | (flags << 2) | value;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:42: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = (flags << 1) || (flags << 2) || value;
    result = (flags << 1) | (flags << 2) | (flags << 4) | value;
    // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:42: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:57: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = (flags << 1) || (flags << 2) || (flags << 4) || value;
}

void assign_to_boolean_parens(unsigned flags, bool value) {
    struct A { bool a = (true | 1); };
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct A { bool a = (true || 1); };
    struct B { union { bool a = (true | 1); }; };
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct B { union { bool a = (true || 1); }; };
    bool result = ((flags << 1) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool result = ((flags << 1) || value);
    bool a = ((flags << 2) | value),
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool a = ((flags << 2) || value),
         b = ((flags << 4) | value),
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: b = ((flags << 4) || value),
         c = ((flags << 8) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: c = ((flags << 8) || value);
    result = ((flags << 1) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = ((flags << 1) || value);
    take(((flags << 1) | value));
    // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: take(((flags << 1) || value));
    result = ((flags << 1) | (flags << 2) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:43: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = ((flags << 1) || (flags << 2) || value);
    result = ((flags << 1) | (flags << 2) | (flags << 4) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:43: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:58: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = ((flags << 1) || (flags << 2) || (flags << 4) || value);
}

void assign_to_boolean_parens2(unsigned flags, bool value) {
    struct A { bool a = ((true | 1)); };
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct A { bool a = ((true || 1)); };
    struct B { union { bool a = ((true | 1)); }; };
    // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct B { union { bool a = ((true || 1)); }; };
    bool result = (((flags << 1) | value));
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool result = (((flags << 1) || value));
    bool a = (((flags << 2) | value)),
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool a = (((flags << 2) || value)),
         b = (((flags << 4) | value)),
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: b = (((flags << 4) || value)),
         c = (((flags << 8) | value));
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: c = (((flags << 8) || value));
    result = (((flags << 1) | value));
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = (((flags << 1) || value));
    take((((flags << 1) | value)));
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: take((((flags << 1) || value)));
    result = (((flags << 1) | (flags << 2) | value));
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:44: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = (((flags << 1) || (flags << 2) || value));
    result = (((flags << 1) | (flags << 2) | (flags << 4) | value));
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:44: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:59: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = (((flags << 1) || (flags << 2) || (flags << 4) || value));
}

// functional cast
void assign_to_boolean_fcast(unsigned flags, bool value) {
    struct A { bool a = bool(true | 1); };
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct A { bool a = bool(true || 1); };
    struct B { union { bool a = bool(true | 1); }; };
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct B { union { bool a = bool(true || 1); }; };
    bool result = bool((flags << 1) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool result = bool((flags << 1) || value);
    bool a = bool((flags << 2) | value),
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool a = bool((flags << 2) || value),
         b = bool((flags << 4) | value),
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: b = bool((flags << 4) || value),
         c = bool((flags << 8) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: c = bool((flags << 8) || value);
    result = bool((flags << 1) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = bool((flags << 1) || value);
    take(bool((flags << 1) | value));
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: take(bool((flags << 1) || value));
    result = bool((flags << 1) | (flags << 2) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:47: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = bool((flags << 1) || (flags << 2) || value);
    result = bool((flags << 1) | (flags << 2) | (flags << 4) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:47: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:62: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = bool((flags << 1) || (flags << 2) || (flags << 4) || value);
}

// C-style cast
void assign_to_boolean_ccast(unsigned flags, bool value) {
    struct A { bool a = (bool)(true | 1); };
    // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct A { bool a = (bool)(true || 1); };
    struct B { union { bool a = (bool)(true | 1); }; };
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct B { union { bool a = (bool)(true || 1); }; };
    bool result = (bool)((flags << 1) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool result = (bool)((flags << 1) || value);
    bool a = (bool)((flags << 2) | value),
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool a = (bool)((flags << 2) || value),
         b = (bool)((flags << 4) | value),
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: b = (bool)((flags << 4) || value),
         c = (bool)((flags << 8) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: c = (bool)((flags << 8) || value);
    result = (bool)((flags << 1) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = (bool)((flags << 1) || value);
    take(bool((flags << 1) | value));
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: take(bool((flags << 1) || value));
    result = (bool)((flags << 1) | (flags << 2) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:49: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = (bool)((flags << 1) || (flags << 2) || value);
    result = (bool)((flags << 1) | (flags << 2) | (flags << 4) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:49: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:64: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = (bool)((flags << 1) || (flags << 2) || (flags << 4) || value);
}

// static_cast
void assign_to_boolean_scast(unsigned flags, bool value) {
    struct A { bool a = static_cast<bool>(true | 1); };
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct A { bool a = static_cast<bool>(true || 1); };
    struct B { union { bool a = static_cast<bool>(true | 1); }; };
    // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: struct B { union { bool a = static_cast<bool>(true || 1); }; };
    bool result = static_cast<bool>((flags << 1) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:50: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool result = static_cast<bool>((flags << 1) || value);
    bool a = static_cast<bool>((flags << 2) | value),
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool a = static_cast<bool>((flags << 2) || value),
         b = static_cast<bool>((flags << 4) | value),
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: b = static_cast<bool>((flags << 4) || value),
         c = static_cast<bool>((flags << 8) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: c = static_cast<bool>((flags << 8) || value);
    result = static_cast<bool>((flags << 1) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = static_cast<bool>((flags << 1) || value);
    take(static_cast<bool>((flags << 1) | value));
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: take(static_cast<bool>((flags << 1) || value));
    result = static_cast<bool>((flags << 1) | (flags << 2) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:60: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = static_cast<bool>((flags << 1) || (flags << 2) || value);
    result = static_cast<bool>((flags << 1) | (flags << 2) | (flags << 4) | value);
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:60: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:75: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: result = static_cast<bool>((flags << 1) || (flags << 2) || (flags << 4) || value);
}


