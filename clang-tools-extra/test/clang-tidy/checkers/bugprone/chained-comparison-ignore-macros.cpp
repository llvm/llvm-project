// RUN: %check_clang_tidy -std=c++98-or-later --extra-arg=-Wno-error=parentheses %s bugprone-chained-comparison %t -- -config="{CheckOptions: {bugprone-chained-comparison.IgnoreMacros: true}}"

#define CHAINED_COMPARE(a, b, c) (a < b < c)

void macro_test(int x, int y, int z) {
    bool result = CHAINED_COMPARE(x, y, z);
}

#define NESTED_LESS(a, b) a < b
#define NESTED_CHAIN(a, b, c) NESTED_LESS(a, b) < c

void nested_macro_test(int x, int y, int z) {
    bool result = NESTED_CHAIN(x, y, z);
}

#define LESS_OP <

void operator_macro_test(int x, int y, int z) {
    bool result = x LESS_OP y LESS_OP z;
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: chained comparison 'v0 < v1 < v2' may generate unintended results

#define PARTIAL_LESS(a, b) a < b

void mixed_macro_test(int x, int y, int z) {
    bool result = PARTIAL_LESS(x, y) < z;
}

void if_macro_test(int x, int y, int z) {
    if (CHAINED_COMPARE(x, y, z)) {}
}

#define LONG_CHAIN_MACRO(v) v[0] < v[1] < v[2] < v[3]

void long_chain_macro_test(int v[4]) {
    bool result = LONG_CHAIN_MACRO(v);
}
