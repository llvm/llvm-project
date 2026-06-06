int add(int a, int b) { return a + b; }
int minus(int a, int b) { return a - b; }
int multiple(int a, int b) { return a * b; }
int divide(int a, int b) {
  if (b == 0)
    return 0;
  return a / b;
}

int main() {
  int a = 16;
  int b = 8;

  for (int i = 1; i < 1000000; i++) {
    add(a, b);
    minus(a, b);
    multiple(a, b);
    divide(a, b);
  }

  return 0;
}
