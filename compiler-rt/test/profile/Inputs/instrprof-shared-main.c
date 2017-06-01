extern int g1, g2;
extern void foo(int n);

int main() {
  int i, j;
  for (i = 0; i < 1000; i++)
    for (j = 0; j < 1000; j++)
      foo(i * j);

  if (g2 - g1 == 280001)
    return 0;
  return 1;
}
