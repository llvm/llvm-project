// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

bool& normal() {
    int a = 100, b = 200;

    a bitor b;
    a bitand b;
    a or_eq b;
    a and_eq b;

    static bool st = false;
    return st;
}

bool bad() noexcept __attribute__((pure)) {
    bool a = true, b = false;
    a bitor b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a or b;
    a bitand b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a and b;
    a or_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a or b;
    a and_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a and b;

    return true;
}

bool global_1 = bad() bitor bad();
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
// CHECK-FIXES: bool global_1 = bad() or bad();
bool global_2 = bad() bitand bad();
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
// CHECK-FIXES: bool global_2 = bad() and bad();

using Boolean = bool;

bool bad_typedef() {
    Boolean a = true, b = false;
    a bitor b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a or b;
    a bitand b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a and b;
    a or_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a or b;
    a and_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a and b;
    return true;
}

bool function_with_possible_side_effects();

void bad_side_effects() {
    bool a = true, b = false;

    a bitor function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a bitand function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    function_with_possible_side_effects() bitor a;
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: function_with_possible_side_effects() or a;

    function_with_possible_side_effects() bitand a;
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: function_with_possible_side_effects() and a;
    a or_eq function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a and_eq function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // count of evaluation with side effect remains the same, so the fixit will be provided
    bool c = true;

    a or function_with_possible_side_effects() bitor c;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a or function_with_possible_side_effects() or c;

    function_with_possible_side_effects() or b bitor c;
    // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: function_with_possible_side_effects() or b or c;

    a and function_with_possible_side_effects() bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:49: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a and function_with_possible_side_effects() and c;

    function_with_possible_side_effects() and b bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:49: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: function_with_possible_side_effects() and b and c;

    // but here the count of evaluation migh be changed - no fix must be provided

    a and_eq function_with_possible_side_effects() and c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a and_eq b and function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a or_eq function_with_possible_side_effects() or c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a or_eq b or function_with_possible_side_effects();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void bad_pointers() {
    bool pointee = false;
    bool* a = &pointee;
    bool* b = &pointee;
    *a bitor *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *a or *b;
    *a bitand *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *a and *b;

    // FIXME: implement fixit for these 2 cases
    // *a or_eq *b;
    // *a and_eq *b;
}

void bad_compound_pointers_as_lhs() {
    bool a = true, b = false;
    bool *pa = &a;
    *pa or_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *pa = *pa or b;
    *pa and_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: *pa = *pa and b;
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
    *wrapper or_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    *wrapper and_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void bad_side_effects_volatile() {
    bool a = true;
    volatile bool b = false;
    bool c = true;

    a bitor b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a bitand b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a or_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a and_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a bitor c) bitor b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a or c) bitor b;

    a bitor c bitor b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a or c bitor b;
}

void bad_side_effects_volatile_typedef() {
    using volatile_bool_t = volatile bool;
    bool a = true;
    volatile_bool_t b = false;
    bool c = true;

    a bitor b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a bitand b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    a or_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    a and_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    (a bitor c) bitor b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (a or c) bitor b;

    a bitor c bitor b;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a or c bitor b;
}

void bad_side_effects_volatile_typedef_pointers() {
    using volatile_bool_t = volatile bool;
    bool pointee = false;
    volatile_bool_t* a = &pointee;
    volatile_bool_t* b = &pointee;
    *a bitor *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    *a bitand *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // FIXME: implement fixit for these 2 cases
    // *a or_eq *b;
    // *a and_eq *b;
}

void bad_side_effects_volatile_typedef_pointers_2() {
    using volatile_bool_t = volatile bool*;
    bool pointee = false;
    volatile_bool_t a = &pointee;
    volatile_bool_t b = &pointee;
    *a bitor *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    *a bitand *b;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes

    // FIXME: implement fixit for these 2 cases
    // *a or_eq *b;
    // *a and_eq *b;
}

void bad_with_priors() {
    bool a = false, b = true, c = true;
    a and b bitor c;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a and (b or c);
    a and b bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&'  [misc-bool-bitwise-operation]
    // CHECK-FIXES: a and b and c;
    a or b bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a or b and c;
    a or b bitor c;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a or b or c;
    b bitor c and a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b or c) and a;

    bool q = (true and false bitor true) and (false bitor true and (false and true bitor false));
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:53: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:84: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool q = (true and (false or true)) and ((false or true) and (false and (true or false)));
}

void bad_with_priors2() {
    bool a = false, b = true, c = true;
    a xor b bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a xor (b and c);
    // braces added in the first change
    a bitor b bitand c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a or (b and c);

    b bitand c xor a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b and c) xor a;

    // braces added in the first change
    b bitand c bitor a;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b and c) or a;
}

template<typename T>
T ident(T val) { return val; }

// cases to check `hasAncestor` works as we expected:
void bad_has_ancestor() {
    bool a = false, b = true, c = true;
    bool d = false;
    d xor (a and b bitand c);
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: d xor (a and b and c);

    a xor ident(b bitand c or a);
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a xor ident(b and c or a);

    a bitor ident(a ? b bitand c : c);
    // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a bitor ident(a ? b and c : c);
}

void bad_with_priors_already_braced() {
    bool a = false, b = true, c = true;
    a and (b bitor c);
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a and (b or c);
    (b bitor c) and a;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b or c) and a;

    bool q = (true and (false bitor true)) and ((false bitor true) and (false and (true bitor false)));
    // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:56: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-3]]:89: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: bool q = (true and (false or true)) and ((false or true) and (false and (true or false)));

    a xor (b bitand c);
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a xor (b and c);

    a bitor (b bitand c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a or (b and c);

    (b bitand c) xor a;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b and c) xor a;

    (b bitand c) bitor a;
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES: :[[@LINE-2]]:18: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|' [misc-bool-bitwise-operation]
    // CHECK-FIXES: (b and c) or a;
}

void bad_with_priors_compound() {
    bool a = false, b = true, c = true;
    a and_eq b or c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a and (b or c);
    a or_eq b or c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a or b or c;
    a and_eq b and c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a and b and c;
    // Braces added because `ParenCompounds` enabled by default
    a or_eq b and c;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a or (b and c);
}

void bad_with_priors_compound_already_braced() {
    bool a = false, b = true, c = true;
    a and_eq (b or c);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-FIXES: a = a and (b or c);
}

void bad_no_fixit() {
    bool b = false;
    normal() or_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '||' for boolean semantics instead of bitwise operator '|=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
    normal() and_eq b;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use logical operator '&&' for boolean semantics instead of bitwise operator '&=' [misc-bool-bitwise-operation]
    // CHECK-MESSAGES-NOT: :[[@LINE-2]]:{{.*}}: note: FIX-IT applied suggested code changes
}

void test_bug_isBooleanBitwise_recursive_classification() {
    bool a = true, b = false, c = true;
    int i = 0;

    bool result1 = (a + b) bitor i;

    bool result2 = (a * b + c) bitor i;
}
