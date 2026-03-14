int g1 = 0;
int g2 = 1;

void foo(int n) {
  if (n % 5 == 0)
    g1++;
  else
    g2++;
}
