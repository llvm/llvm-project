! RUN: %flang -dM -E -o - %s | FileCheck %s

! Variadic macro
#define FOO1(X, Y, ...) bar(bar(X, Y), __VA_ARGS__)
! CHECK: #define FOO1(X, Y, ...) bar(bar(X, Y), __VA_ARGS__)

! Macro with an unused parameter
#define FOO2(X, Y, Z) (X + Z)
! CHECK: #define FOO2(X, Y, Z) (X + Z)
