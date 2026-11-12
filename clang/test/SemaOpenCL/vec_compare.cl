// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

typedef __attribute__((ext_vector_type(2)))  unsigned int uint2;
typedef __attribute__((ext_vector_type(2)))  int int2;

void unsignedCompareOps(void)
{
  uint2 A, B;
  int2 result = A != B;
}

