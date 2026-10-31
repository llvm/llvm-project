// RUN: %clang_cc1 -print-dependency-directives-minimized-source %s > %t
// RUN: %clang_cc1 -E -dM %t | FileCheck %s

// Object-like macro: losing the space would turn this into an invalid function-like macro definition.
#define ENABLED (1)
#if ENABLED
#define OBJECT_OK
#endif
// CHECK-DAG: #define OBJECT_OK

// Function-like macro: no space may be inserted between the macro name and the opening parenthesis.
#define ID(x) (x)
#define NO_ARGS() (1)
#if ID(1) && !ID(0) && NO_ARGS()
#define FUNCTION_OK
#endif
// CHECK-DAG: #define FUNCTION_OK
