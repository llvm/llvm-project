// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null



static int foo(int);

static int foo(C)
char C;
{
  return C;
}

void test() {
  foo(7);
}
