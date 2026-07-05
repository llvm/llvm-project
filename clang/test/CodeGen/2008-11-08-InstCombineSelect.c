// RUN: %clang_cc1 %s -emit-llvm -O2 -o -
// PR3028

int g_187;
int g_204;
int g_434;

int func_89 (void)
{
  return 1;
}

void func_20 (int p_22)
{
  if (1 & p_22 | g_204 & (1 < g_187) - func_89 ())
    g_434 = 1;
}
