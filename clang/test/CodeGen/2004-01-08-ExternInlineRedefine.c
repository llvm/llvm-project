// RUN: %clang_cc1 -std=gnu89 -emit-llvm %s  -o /dev/null


extern __inline long int
__strtol_l (int a)
{
  return 0;
}

long int
__strtol_l (int a)
{
  return 0;
}
