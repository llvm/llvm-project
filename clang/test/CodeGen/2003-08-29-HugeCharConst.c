// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

void foo() {
  unsigned char int_latin1[] = "f\200\372b\200\343\200\340";
}
