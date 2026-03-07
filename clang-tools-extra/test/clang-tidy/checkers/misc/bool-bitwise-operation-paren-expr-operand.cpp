// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

bool& normal() {
    int a = 100, b = 200;

    (a) | (b);
    (a) & (b);
    (a) |= (b);
    (a) &= (b);

    static bool st = false;
    return st;
}

bool bad() noexcept __attribute__((pure)) {
    bool a = true, b = false;
    (a) | (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) || (b);
    (a) & (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) && (b);
    (a) |= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) = (a) || (b);
    (a) &= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) = (a) && (b);

    return true;
}

bool global_1 = (bad()) | (bad());
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
// CHECK-FIXES: bool global_1 = (bad()) || (bad());
bool global_2 = (bad()) & (bad());
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
// CHECK-FIXES: bool global_2 = (bad()) && (bad());

using Boolean = bool;

bool bad_typedef() {
    Boolean a = true, b = false;
    (a) | (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) || (b);
    (a) & (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) && (b);
    (a) |= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) = (a) || (b);
    (a) &= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) = (a) && (b);
    return true;
}

bool function_with_possible_side_effects();

void bad_side_effects() {
    bool a = true, b = false;

    (a) | (function_with_possible_side_effects());
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a) & (function_with_possible_side_effects());
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (function_with_possible_side_effects()) | (a);
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (function_with_possible_side_effects()) || (a);

    (function_with_possible_side_effects()) & (a);
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (function_with_possible_side_effects()) && (a);
    (a) |= (function_with_possible_side_effects());
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a) &= (function_with_possible_side_effects());
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // count of evaluation with side effect remains the same, so the fixit will be provided
    bool c = true;

    (a) || (function_with_possible_side_effects()) | (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) || (function_with_possible_side_effects()) || (c);

    (function_with_possible_side_effects()) || (b) | (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (function_with_possible_side_effects()) || (b) || (c);

    (a) && (function_with_possible_side_effects()) & (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) && (function_with_possible_side_effects()) && (c);

    (function_with_possible_side_effects()) && (b) & (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:52: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (function_with_possible_side_effects()) && (b) && (c);

    // but here the count of evaluation migh be changed - no fix must be provided

    (a) &= (function_with_possible_side_effects()) && (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a) &= (b) && (function_with_possible_side_effects());
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a) |= (function_with_possible_side_effects()) || (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a) |= (b) || (function_with_possible_side_effects());
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void bad_pointers() {
    bool pointee = false;
    bool* a = &pointee;
    bool* b = &pointee;
    (*a) | (*b);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (*a) || (*b);
    (*a) & (*b);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (*a) && (*b);

    // FIXME: implement fixit for these 2 cases
    // (*a) |= (*b);
    // (*a) &= (*b);
}

void bad_compound_pointers_as_lhs() {
    bool a = true, b = false;
    bool *pa = &a;
    (*pa) |= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (*pa) = (*pa) || (b);
    (*pa) &= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (*pa) = (*pa) && (b);
}

class BoolWrapper {
    bool value;
public:
    BoolWrapper(bool v) : value(v) {}
    bool& operator*() { return value; }
};

void bad_compound_user_defined_dereference_as_lhs() {
    bool b = false;
    BoolWrapper wrapper(true);
    (*wrapper) |= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (*wrapper) &= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void bad_side_effects_volatile() {
    bool a = true;
    volatile bool b = false;
    bool c = true;

    (a) | (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (a) & (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a) |= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (a) &= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    ((a) | (c)) | (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: ((a) || (c)) | (b);

    (a) | (c) | (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) || (c) | (b);
}

void bad_side_effects_volatile_typedef() {
    using volatile_bool_t = volatile bool;
    bool a = true;
    volatile_bool_t b = false;
    bool c = true;

    (a) | (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (a) & (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a) |= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (a) &= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    ((a) | (c)) | (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: ((a) || (c)) | (b);

    (a) | (c) | (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) || (c) | (b);
}

void bad_side_effects_volatile_typedef_pointers() {
    using volatile_bool_t = volatile bool;
    bool pointee = false;
    volatile_bool_t* a = &pointee;
    volatile_bool_t* b = &pointee;
    (*a) | (*b);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (*a) & (*b);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // FIXME: implement fixit for these 2 cases
    // (*a) |= (*b);
    // (*a) &= (*b);
}

void bad_side_effects_volatile_typedef_pointers_2() {
    using volatile_bool_t = volatile bool*;
    bool pointee = false;
    volatile_bool_t a = &pointee;
    volatile_bool_t b = &pointee;
    (*a) | (*b);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (*a) & (*b);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // FIXME: implement fixit for these 2 cases
    // (*a) |= (*b);
    // (*a) &= (*b);
}

void bad_with_priors() {
    bool a = false, b = true, c = true;
    (a) && (b) | (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) && ((b) || (c));
    (a) && (b) & (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) && (b) && (c);
    (a) || (b) & (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) || (b) && (c);
    (a) || (b) | (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) || (b) || (c);
    (b) | (c) && (a);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: ((b) || (c)) && (a);

    bool q = ((true) && ((false) | (true))) && (((false) | (true)) && ((false) && ((true) | (false))));
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:58: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:91: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool q = ((true) && ((false) || (true))) && (((false) || (true)) && ((false) && ((true) || (false))));
}

void bad_with_priors2() {
    bool a = false, b = true, c = true;
    (a) ^ (b) & (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) ^ ((b) && (c));
    // braces added in the first change
    (a) | (b) & (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) || ((b) && (c));

    (b) & (c) ^ (a);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: ((b) && (c)) ^ (a);

    // braces added in the first change
    (b) & (c) | (a);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: ((b) && (c)) || (a);
}

template<typename T>
T ident(T val) { return val; }

// cases to check `hasAncestor` works as we expected:
void bad_has_ancestor() {
    bool a = false, b = true, c = true;
    bool d = false;
    (d) ^ ((a) && (b) & (c));
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (d) ^ ((a) && (b) && (c));

    (a) ^ (ident((b) & (c) || (a)));
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) ^ (ident((b) && (c) || (a)));

    (a) | (ident((a) ? (b) & (c) : (c)));
    // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) | (ident((a) ? (b) && (c) : (c)));
}

void bad_with_priors_already_braced() {
    bool a = false, b = true, c = true;
    (a) && ((b) | (c));
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) && ((b) || (c));
    ((b) | (c)) && (a);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: ((b) || (c)) && (a);

    bool q = ((true) && ((false) | (true))) && (((false) | (true)) && ((false) && ((true) | (false))));
    // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:58: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:91: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool q = ((true) && ((false) || (true))) && (((false) || (true)) && ((false) && ((true) || (false))));

    (a) ^ ((b) & (c));
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) ^ ((b) && (c));

    (a) | ((b) & (c));
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) || ((b) && (c));

    ((b) & (c)) ^ (a);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: ((b) && (c)) ^ (a);

    ((b) & (c)) | (a);
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: ((b) && (c)) || (a);
}

void bad_with_priors_compound() {
    bool a = false, b = true, c = true;
    (a) &= (b) || (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) = (a) && ((b) || (c));
    (a) |= (b) || (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) = (a) || (b) || (c);
    (a) &= (b) && (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) = (a) && (b) && (c);
    // Braces added because `ParenCompounds` enabled by default
    (a) |= (b) && (c);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) = (a) || ((b) && (c));
}

void bad_with_priors_compound_already_braced() {
    bool a = false, b = true, c = true;
    (a) &= ((b) || (c));
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) = (a) && ((b) || (c));
}

void bad_no_fixit() {
    bool b = false;
    (normal()) |= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (normal()) &= (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

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
    IDENT((a) |) (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (a) IDENT(& (b));
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    IDENT((a) |=) (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    (a) IDENT(&= (b));
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // change operator - GOOD
    IDENT((a)) | (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: IDENT((a)) || (b);
    (a) & IDENT((b));
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) && IDENT((b));
    IDENT((a)) & IDENT((b));
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: IDENT((a)) && IDENT((b));

    // insert `)` - BAD
    bool c = true, e = false;
    (a) && (b) | IDENT((c) &&) (e);
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // insert `)` - GOOD
    (a) && (b) | (c) IDENT(&& (e));
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a) && ((b) || (c)) IDENT(&& (e));

    // insert `(` - BAD
    (a) IDENT(&& (b)) | (c) && (e);
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // insert `(` - GOOD
    IDENT((a) &&) (b) | (c) && (e);
    // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: IDENT((a) &&) ((b) || (c)) && (e);

    bool ab = false;

    // FIXME
    // 
    // // insert ` = a` - BAD
    // (CAT(a, b)) &= (b);

    // insert ` = a`- GOOD
    (b) &= (CAT(a, b));
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b) = (b) && (CAT(a, b));
}

void bad_in_macro_fixit() {
    bool a = true, b = false;

    // FIXME: implement fixit for all of these cases
    
    (a) MY_OR (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    (a) MY_AND (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    (a) MY_OR_ASSIGN (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    (a) MY_AND_ASSIGN (b);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    IDENT((a) &= (b));
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
}

template<typename T>
void good_in_unreachable_template(T a, T b) {
    (a) | (b);
    (a) & (b);
    (a) |= (b);
    (a) &= (b);
}

template<typename T>
int bad_in_template(T a, T b) {
    bool c = false;
    // FIXME: at least warning should be provided in these cases
    // (a) | (b);
    // (a) & (b);
    // (a) |= (b);
    // (a) &= (b);
    // (c) &= (a);
    return 0;
}

template<typename T>
int bad_in_template_lambda_captured(T a, T b) {
    [=] mutable {
        bool c = false;
        // FIXME: at least warning should be provided in these cases
        // (a) | (b);
        // (a) & (b);
        // (a) |= (b);
        // (b) &= (a);
    }();
    return 0;
}

int dummy = (bad_in_template((true), (false))) + (bad_in_template_lambda_captured((false), (true)));

void test_bug_isBooleanBitwise_recursive_classification() {
    bool a = true, b = false, c = true;
    int i = 0;

    bool result1 = ((a) + (b)) | (i);

    bool result2 = ((a) * (b) + (c)) | (i);
}
