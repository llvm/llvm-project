// RUN: %clang_cc1 -emit-llvm %s -o %t

void f(void* arg);
void g(void) {
  __attribute__((cleanup(f))) void *g;
}

