int loop(int n);

void foo() {
  int x = 0;
  int y = x + x;
  int ret = -1;
  ret = loop(x);
}
