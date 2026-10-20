void test(int a, int b, int c, int d) {
  if ((a && b) || (c && d))
    ;
  if (b && c)
    ;
}

int main() {
  test(1, 1, 1, 1);
  test(1, 1, 0, 1);
  test(0, 0, 1, 0);
  test(0, 0, 1, 1);
  return 0;
}
