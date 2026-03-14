// RUN: %clang_cc1 -emit-llvm -Wno-int-conversion %s -o -

int test(void* i)
{
  return (int)i;
}

int test2(void) {
  float x[2];
  return x;
}

