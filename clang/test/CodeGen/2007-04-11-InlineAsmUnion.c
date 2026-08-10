// RUN: %clang_cc1 %s -emit-llvm -o -

union U { int x; float p; };
void foo(void) {
  union U bar;
  __asm__ volatile("foo %0\n" : "=r"(bar));
}
