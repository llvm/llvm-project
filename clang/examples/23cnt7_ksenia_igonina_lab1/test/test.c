int var1 = 0;

int foo(int a, int b) {
  static int var2 = 0;
  int var3 = 123;
  ++var2;
  return a + b + var1 + var2 + var3;
}
