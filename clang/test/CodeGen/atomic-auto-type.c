// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s > /dev/null

// This is a regression test for handling of __auto_type inside _Atomic.
// Previously this could lead to an undeduced AutoType escaping into
// ASTContext::getTypeInfoImpl and causing an assertion failure.

void f(double x) {
  __auto_type _Atomic xa = x;
  _Atomic __auto_type ax = x;
}
