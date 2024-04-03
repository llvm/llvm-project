! RUN: %flang -dM -E -o - %s | FileCheck %s

! Variadic macro
#define FOO1(X, Y, ...) bar(bar(X, Y), __VA_ARGS__)
! CHECK: #define FOO1(A, B, ...) bar(bar(A, B), __VA_ARGS__)

! Macro parameter names are synthesized, starting from 'A', B', etc.
! Make sure the generated names do not collide with existing identifiers.
#define FOO2(X, Y) (A + X + C + Y)
! CHECK: #define FOO2(B, D) (A + B + C + D)
