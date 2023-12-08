// RUN: %clang_cc1 %s -emit-llvm -o -

void f(const char*);

void g() {
  f("hello");
}
