// RUN: %clang_cc1 -emit-llvm %s -o -

int test(void* i)
{
  return (int)i;
}

// rdar://6093986
int test2(void) {
  float x[2];
  return x;
}

