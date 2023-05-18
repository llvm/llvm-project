// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s

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
