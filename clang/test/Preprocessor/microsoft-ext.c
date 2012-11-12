// RUN: %clang_cc1 -E -fms-compatibility %s -o %t
// RUN: FileCheck %s < %t

# define M2(x, y) x + y
# define P(x, y) {x, y}
# define M(x, y) M2(x, P(x, y))
M(a, b) // CHECK: a + {a, b}

// Regression test for PR13924
#define GTEST_CONCAT_TOKEN_(foo, bar) GTEST_CONCAT_TOKEN_IMPL_(foo, bar)
#define GTEST_CONCAT_TOKEN_IMPL_(foo, bar) foo ## bar

#define GMOCK_INTERNAL_COUNT_AND_2_VALUE_PARAMS(p0, p1) P2

#define GMOCK_ACTION_CLASS_(name, value_params)\
    GTEST_CONCAT_TOKEN_(name##Action, GMOCK_INTERNAL_COUNT_##value_params)

#define ACTION_TEMPLATE(name, template_params, value_params)\
class GMOCK_ACTION_CLASS_(name, value_params) {\
}

ACTION_TEMPLATE(InvokeArgument,
                HAS_1_TEMPLATE_PARAMS(int, k),
                AND_2_VALUE_PARAMS(p0, p1));
