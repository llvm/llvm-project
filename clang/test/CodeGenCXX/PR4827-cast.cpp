// RUN: %clang_cc1 -emit-llvm -o - %s
struct A;
struct B;
extern A *f();
void a() { (B *) f(); }
