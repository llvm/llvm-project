// RUN: %clang_cc1 -print-dependency-directives-minimized-source %s > %t
// RUN: %clang_cc1 -E -dM %t | FileCheck %s

// The conditional defines below all rely on the printer preserving the macro
// definitions and #if conditions accurately.

#define ADD(x, y) ((x) + (y))
#define SHL(x, n) ((x) << (n))
#define HAS(x) (x)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define BAR() 42

#if ADD(1, 2) > 0
#define POSITIVE_SUM
// CHECK-DAG: #define POSITIVE_SUM
#endif

#if ADD(1, 2) == 3 && SHL(1, 3) == 8
#define MATH_OK
// CHECK-DAG: #define MATH_OK
#endif

#if HAS(0) || HAS(1)
#define HAS_EITHER
// CHECK-DAG: #define HAS_EITHER
#endif

#if !ADD(0, 0)
#define NEGATED
// CHECK-DAG: #define NEGATED
#endif

#if 1 + HAS(2)
#define LEADING_NUM
// CHECK-DAG: #define LEADING_NUM
#endif

#if BAR() + BAR() == 84
#define EMPTY_ARGS
// CHECK-DAG: #define EMPTY_ARGS
#endif

#if MIN(1, 2) == 1
#define MIN_OK
// CHECK-DAG: #define MIN_OK
#endif

#if HAS(1) - HAS(2) < 0
#define SUB_OK
// CHECK-DAG: #define SUB_OK
#endif

#if HAS(0xff) & 0xf0
#define BITWISE
// CHECK-DAG: #define BITWISE
#endif

#if HAS(1) << 2 | HAS(2)
#define SHIFT_OR
// CHECK-DAG: #define SHIFT_OR
#endif

#if defined(FOO) && ADD(1, 2) + ADD(3, 4) > 5
#define COMPLEX_BUT_FALSE
// CHECK-NOT: #define COMPLEX_BUT_FALSE
#endif
