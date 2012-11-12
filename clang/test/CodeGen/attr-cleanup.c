// RUN: %clang_cc1 -emit-llvm %s -o %t

// <rdar://problem/6827047>
void f(void* arg);
void g() {
  __attribute__((cleanup(f))) void *g;
}

