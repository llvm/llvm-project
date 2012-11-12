// RUN: %clang_cc1 -emit-llvm %s -o - | grep llvm.global.annotations
// RUN: %clang_cc1 -emit-llvm %s -o - | grep llvm.var.annotation | count 3

/* Global variable with attribute */
int X __attribute__((annotate("GlobalValAnnotation")));

/* Function with attribute */
int foo(int y) __attribute__((annotate("GlobalValAnnotation")))
               __attribute__((noinline));

int foo(int y __attribute__((annotate("LocalValAnnotation")))) {
  int x __attribute__((annotate("LocalValAnnotation")));
  x = 34;
  return y + x;
}

int main() {
  static int a __attribute__((annotate("GlobalValAnnotation")));
  a = foo(2);
  return 0;
}
