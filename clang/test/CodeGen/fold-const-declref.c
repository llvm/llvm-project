// RUN: %clang_cc1 -verify -emit-llvm-only

// PR7242: Check that this doesn't crash.
int main(void)
{
  int __negative = 1;
  const int __max = __negative && 0 ;
  __max / 0;
}
