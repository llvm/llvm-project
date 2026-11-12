// RUN: %clang_cc1 -DFIRST_WAY -E -dM %s | FileCheck --strict-whitespace %s
// RUN: %clang_cc1 -DFIRST_WAY -fms-compatibility -E -dM %s | FileCheck --strict-whitespace %s
// RUN: %clang_cc1 -E -dM %s | FileCheck --strict-whitespace %s
// RUN: %clang_cc1 -fms-compatibility -E -dM %s | FileCheck --strict-whitespace %s

// If the assert macro is defined in MS compatibility mode in C, we
// automatically inject a macro definition for static_assert. Test that the
// macro is not added if there is already a definition of static_assert to
// ensure that we don't re-define the macro in the event the Microsoft assert.h
// header starts to define the macro some day (or the user defined their own
// macro with the same name). Test that the order of the macro definitions does
// not matter to the behavior.

#ifdef FIRST_WAY
#define static_assert 12
#define assert
#else
#define assert
#define static_assert 12
#endif

CHECK: #define static_assert 12

