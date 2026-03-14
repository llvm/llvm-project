// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


union X {
  void *B;
};

union X foo(void) {
  union X A;
  A.B = (void*)123;
  return A;
}
