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

// Consecutive ## operators can be valid when pasting empty arguments.
#define PASTE1(X) X ## X X ## ## X X ## X
e:PASTE1()
// CHECK: e:
// CHECK-NOT: ##

// The leading placemarker can also supply the LHS of a nonempty paste.
#define PASTE_LEADING(X) X ## X X ## ## 2
f:PASTE_LEADING()
// CHECK: f:2
