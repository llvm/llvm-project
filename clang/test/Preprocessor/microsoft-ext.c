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

// Regression test for PR43282; check that we match MSVC's failure to unpack
// __VA_ARGS__ unless forwarded through another macro.
#define THIRD_ARGUMENT(A, B, C, ...) C
#define TEST(...) THIRD_ARGUMENT(__VA_ARGS__, 1, 2)
#define COMBINE(...) __VA_ARGS__
#define WRAPPED_TEST(...) COMBINE(THIRD_ARGUMENT(__VA_ARGS__, 1, 2))
// Check that we match MSVC's failure to unpack __VA_ARGS__, unless forwarded
// through another macro
auto packed = TEST(,);
auto unpacked = WRAPPED_TEST(,);
// CHECK: auto packed = 2;
// CHECK: auto unpacked = 1;

// This tests compatibility with behaviour needed for type_traits in VS2012
// Test based on _VARIADIC_EXPAND_0X macros in xstddef of VS2012
#define _COMMA ,

#define MAKER(_arg1, _comma, _arg2)                                            \
  void func(_arg1 _comma _arg2) {}
#define MAKE_FUNC(_makerP1, _makerP2, _arg1, _comma, _arg2)                    \
  _makerP1##_makerP2(_arg1, _comma, _arg2)

MAKE_FUNC(MAK, ER, int a, _COMMA, int b);
// CHECK: void func(int a , int b) {}

#define macro(a, b) (a - b)
void function(int a);
#define COMMA_ELIDER(...) \
  macro(x, __VA_ARGS__); \
  function(x, __VA_ARGS__);
COMMA_ELIDER();
// CHECK: (x - );
// CHECK: function(x);
