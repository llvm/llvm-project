// RUN: %clang_cc1 %s -o %t -emit-llvm -verify
// expected-no-diagnostics
// PR4289

struct funcptr {
  int (*func)();
};

static int func(f)
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
