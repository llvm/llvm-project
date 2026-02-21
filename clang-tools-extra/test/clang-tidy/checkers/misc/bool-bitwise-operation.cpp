// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

bool& normal() {
    int a = 100, b = 200;

    a | b;
    a & b;
    a |= b;
    a &= b;

    static bool st = false;
    return st;
}

bool bad() noexcept __attribute__((pure)) {
    bool a = true, b = false;
    a | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || b;
    a & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && b;
    a |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b;
    a &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b;

    return true;
}

bool global_1 = bad() | bad();
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
// CHECK-FIXES: bool global_1 = bad() || bad();
bool global_2 = bad() & bad();
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
// CHECK-FIXES: bool global_2 = bad() && bad();

using Boolean = bool;

bool bad_typedef() {
    Boolean a = true, b = false;
    a | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || b;
    a & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && b;
    a |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b;
    a &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b;
    return true;
}

bool function_with_possible_side_effects();

void bad_side_effects() {
    bool a = true, b = false;

    a | function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a & function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    function_with_possible_side_effects() | a;
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: function_with_possible_side_effects() || a;

    function_with_possible_side_effects() & a;
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: function_with_possible_side_effects() && a;
    a |= function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a &= function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // count of evaluation with side effect remains the same, so the fixit will be provided
    bool c = true;

    a || function_with_possible_side_effects() | c;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || function_with_possible_side_effects() || c;

    function_with_possible_side_effects() || b | c;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: function_with_possible_side_effects() || b || c;

    a && function_with_possible_side_effects() & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && function_with_possible_side_effects() && c;

    function_with_possible_side_effects() && b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: function_with_possible_side_effects() && b && c;

    // but here the count of evaluation migh be changed - no fix must be provided

    a &= function_with_possible_side_effects() && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a &= b && function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= function_with_possible_side_effects() || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= b || function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void bad_pointers() {
    bool pointee = false;
    bool* a = &pointee;
    bool* b = &pointee;
    *a | *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *a || *b;
    *a & *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *a && *b;

    // FIXME: implement fixit for these 2 cases
    // *a |= *b;
    // *a &= *b;
}

void bad_compound_pointers_as_lhs() {
    bool a = true, b = false;
    bool *pa = &a;
    *pa |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *pa = *pa || b;
    *pa &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *pa = *pa && b;
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
    *wrapper |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    *wrapper &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void bad_side_effects_volatile() {
    bool a = true;
    volatile bool b = false;
    bool c = true;

    a | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a | c) | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:13: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a || c) | b;

    a | c | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || c | b;
}

void bad_side_effects_volatile_typedef() {
    using volatile_bool_t = volatile bool;
    bool a = true;
    volatile_bool_t b = false;
    bool c = true;

    a | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a & b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a | c) | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:13: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a || c) | b;

    a | c | b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || c | b;
}

void bad_side_effects_volatile_typedef_pointers() {
    using volatile_bool_t = volatile bool;
    bool pointee = false;
    volatile_bool_t* a = &pointee;
    volatile_bool_t* b = &pointee;
    *a | *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    *a & *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // FIXME: implement fixit for these 2 cases
    // *a |= *b;
    // *a &= *b;
}

void bad_side_effects_volatile_typedef_pointers_2() {
    using volatile_bool_t = volatile bool*;
    bool pointee = false;
    volatile_bool_t a = &pointee;
    volatile_bool_t b = &pointee;
    *a | *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    *a & *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // FIXME: implement fixit for these 2 cases
    // *a |= *b;
    // *a &= *b;
}

void bad_with_priors() {
    bool a = false, b = true, c = true;
    a && b | c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && (b || c);
    a && b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && b && c;
    a || b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || b && c;
    a || b | c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || b || c;
    b | c && a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b || c) && a;

    bool q = (true && false | true) && (false | true && (false && true | false));
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:47: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:72: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool q = (true && (false || true)) && ((false || true) && (false && (true || false)));
}

void bad_with_priors2() {
    bool a = false, b = true, c = true;
    a ^ b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a ^ (b && c);
    // braces added in the first change
    a | b & c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || (b && c);

    b & c ^ a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b && c) ^ a;

    // braces added in the first change
    b & c | a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:11: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b && c) || a;
}

template<typename T>
T ident(T val) { return val; }

// cases to check `hasAncestor` works as we expected:
void bad_has_ancestor() {
    bool a = false, b = true, c = true;
    bool d = false;
    d ^ (a && b & c);
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: d ^ (a && b && c);

    a ^ ident(b & c || a);
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a ^ ident(b && c || a);

    a | ident(a ? b & c : c);
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a | ident(a ? b && c : c);
}

void bad_with_priors_already_braced() {
    bool a = false, b = true, c = true;
    a && (b | c);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a && (b || c);
    (b | c) && a;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b || c) && a;

    bool q = (true && (false | true)) && ((false | true) && (false && (true | false)));
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:50: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:77: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool q = (true && (false || true)) && ((false || true) && (false && (true || false)));

    a ^ (b & c);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a ^ (b && c);

    a | (b & c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a || (b && c);

    (b & c) ^ a;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b && c) ^ a;

    (b & c) | a;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:13: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b && c) || a;
}

void bad_with_priors_compound() {
    bool a = false, b = true, c = true;
    a &= b || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && (b || c);
    a |= b || c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || b || c;
    a &= b && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && b && c;
    // Braces added because `ParenCompounds` enabled by default
    a |= b && c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a || (b && c);
}

void bad_with_priors_compound_already_braced() {
    bool a = false, b = true, c = true;
    a &= (b || c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a && (b || c);
}

void bad_no_fixit() {
    bool b = false;
    normal() |= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    normal() &= b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void test_bug_isBooleanBitwise_recursive_classification() {
    bool a = true, b = false, c = true;
    int i = 0;

    bool result1 = (a + b) | i;

    bool result2 = (a * b + c) | i;
}
