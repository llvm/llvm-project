// RUN: %clang_cc1 %s -emit-llvm-only -verify
// expected-no-diagnostics

struct A {};
struct B : A {};
void a(const A& x = B());
void b() { a(); }
