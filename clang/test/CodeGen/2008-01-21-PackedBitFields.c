// RUN: %clang_cc1 %s -emit-llvm -o -

typedef double Al1Double __attribute__((aligned(1)));
struct x { int a:23; Al1Double v; };
struct x X = { 5, 3.0 };
double foo(void) { return X.v; }

