// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

int strcmp(const char *s1, const char *s2);

int test(char *X) {
  /* LLVM-GCC used to emit:
     %.LC0 = internal global [3 x sbyte] c"\1F\FFFFFF8B\00"
   */
  return strcmp(X, "\037\213");
}
