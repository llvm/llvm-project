// RUN: not %clang_cc1 -triple i686-windows-msvc -emit-llvm -o /dev/null %s  2>&1 | FileCheck %s

// PR28299
// CHECK: error: cannot compile this forwarded non-trivially copyable parameter yet

class A {
  A(const A &);
};
typedef void (*fptr_t)(A);
fptr_t fn1() { return [](A) {}; }

