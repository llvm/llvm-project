
int factorProduct(int a, int b) {
  int result = 1;
  int min = a > b ? b : a;
  for (int i = 2; i < min; ++i) {
    bool is_a = a % i == 0;
    bool is_b = b % i == 0;
    if (is_a && is_b)
      result *= i;
  }
  return result;
}
