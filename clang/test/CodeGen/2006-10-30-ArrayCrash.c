// RUN: %clang_cc1 -O3 -emit-llvm -o - %s
// PR954, PR911

extern void foo(void);

struct S {
  short        f1[3];
  unsigned int f2 : 1;
};

void bar(void)
{
  struct S *A;

  if (A->f2)
    foo();
}
