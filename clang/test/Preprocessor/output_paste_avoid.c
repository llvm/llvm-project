// RUN: %clang_cc1 -E %s -o - | FileCheck -strict-whitespace %s


#define y(a) ..a
A: y(.)
// This should print as ".. ." to avoid turning into ...
// CHECK: A: .. .

#define X 0 .. 1
B: X
// CHECK: B: 0 .. 1

#define DOT .
C: ..DOT
// CHECK: C: .. .


#define PLUS +
#define EMPTY
#define f(x) =x=
D: +PLUS -EMPTY- PLUS+ f(=)
// CHECK: D: + + - - + + = = =


#define test(x) L#x
E: test(str)
// Should expand to L "str" not L"str"
// CHECK: E: L "str"

// Should avoid producing >>=.
#define equal =
F: >>equal
// CHECK: F: >> =
