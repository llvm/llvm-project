// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s
// RUN: %clang_cc1 -E -x c++ %s | FileCheck --strict-whitespace %s

#define FOO(X) X ## Y
a:FOO()
// CHECK: a:Y

#define FOO2(X) Y ## X
b:FOO2()
// CHECK: b:Y

#define FOO3(X) X ## Y ## X ## Y ## X ## X
c:FOO3()
// CHECK: c:YY

#define FOO4(X, Y) X ## Y
d:FOO4(,FOO4(,))
// CHECK: d:FOO4

// Consecutive ## operators need not make an expansion invalid: the
// placemarkers from the empty arguments allow this to expand to nothing.
#define PASTE1(X) X ## X X ## ## X X ## X
before PASTE1() after
// CHECK: before after

// The leading placemarker also supplies an operand when the final token is
// nonempty.
#define PASTE_LEADING(X) X ## X X ## ## 2
e:PASTE_LEADING()
// CHECK: e:2

// Each leading placemarker can supply an operand for a surviving operator.
#define PASTE_TWO_LEADING(X) X X X ## ## ## 2
f:PASTE_TWO_LEADING()
// CHECK: f:2

// A ## supplied by an argument is a token, not a paste operator.
#define IDENTITY(X) X
g:IDENTITY(##)
// CHECK: g:##
