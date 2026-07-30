// RUN: %clang_cc1 %s -emit-llvm -o -

void foo(void *ptr, int test) {
  (ptr - ((void *) test + 0x2000));
}
