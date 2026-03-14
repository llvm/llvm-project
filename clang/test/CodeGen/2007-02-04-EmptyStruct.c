// RUN: %clang_cc1 %s -O3 -emit-llvm -o -
// PR1175

struct empty { };

void foo(struct empty *p) {
   p++;
}

