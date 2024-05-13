// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

/* In this testcase, the return value of foo() is being promoted to a register
 * which breaks stuff
 */
int printf(const char * restrict format, ...);

union X { char X; void *B; int a, b, c, d;};

union X foo(void) {
  union X Global;
  Global.B = (void*)123;   /* Interesting part */
  return Global;
}

int main(void) {
  union X test = foo();
  printf("0x%p", test.B);
}
