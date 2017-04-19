#include <stdio.h>

#if defined(__arm__) || defined(__aarch64__)
// Clang does not accept regparm attribute on these platforms.
// Fortunately, the default calling convention passes arguments in registers
// anyway.
#define REGPARM(N)
#else
#define REGPARM(N) __attribute__((regparm(N)))
#endif

struct bar {
  int m1;
  int m2;
};

void f1(int a, struct bar *b) __attribute__((noinline)) REGPARM(2);
void f1(int a, struct bar *b)
{
  b->m2 = b->m1 + a; // set breakpoint here
}

void f2(struct bar *b) __attribute__((noinline)) REGPARM(1);
void f2(struct bar *b)
{
  int c = b->m2;
  printf("%d\n", c); // set breakpoint here
}

float f3() __attribute__((noinline));
float f3() {
  return 3.14f;
}

int main()
{
  struct bar myBar = { 3, 4 };
  f1(2, &myBar);
  f2(&myBar);

  float f = f3();
  printf("%f\n", f); // set breakpoint here
  return 0;
}
