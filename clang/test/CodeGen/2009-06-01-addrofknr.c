// RUN: %clang_cc1 %s -o %t -emit-llvm -verify -std=c89
// PR4289

struct funcptr {
  int (*func)();
};

static int func(f) // expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C2x}}
  void *f;
{
  return 0;
}

int
main(int argc, char *argv[])
{
  struct funcptr fp;

  fp.func = &func;
  fp.func = func;
}
