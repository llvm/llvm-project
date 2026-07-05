// RUN: %clang_cc1 -E -dM %s -o - | FileCheck %s -strict-whitespace

// Space at end even without expansion tokens
// CHECK: #define A(x) 
#define A(x)

// Space before expansion list.
// CHECK: #define B(x,y) x y
#define B(x,y)x y

// No space in argument list.
// CHECK: #define C(x,y) x y
#define C(x, y) x y

// No paste avoidance.
// CHECK: #define D() ..
#define D() ..

// Simple test.
// CHECK: #define E .
// CHECK: #define F X()Y
#define E .
#define F X()Y

// gcc prints macros at end of translation unit, so last one wins.
// CHECK: #define G 2
#define G 1
#undef G
#define G 2

// Variadic macros of various sorts. PR5699

// CHECK: H(x,...) __VA_ARGS__
#define H(x, ...) __VA_ARGS__
// CHECK: I(...) __VA_ARGS__
#define I(...) __VA_ARGS__
// CHECK: J(x...) __VA_ARGS__
#define J(x ...) __VA_ARGS__
