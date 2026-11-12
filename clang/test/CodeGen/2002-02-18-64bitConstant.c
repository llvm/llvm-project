// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

/* GCC wasn't handling 64 bit constants right fixed */

int printf(const char * restrict format, ...);

int main(void) {
  long long Var = 123455678902ll;
  printf("%lld\n", Var);
}
