// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

// expected-no-diagnostics

float __attribute__((overloadable)) acos(float);

typedef float float4 __attribute__((ext_vector_type(4)));
int printf(__constant const char* st, ...);

void test(void)
{
  float4 a;
  printf("%8.4v4hlf\n", a);
}
