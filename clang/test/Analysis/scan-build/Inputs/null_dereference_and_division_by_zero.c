int test(int x) {
  if (x) {
    int *p = 0;
    return *p; // Null dereference.
  } else {
    return 1 / x; // Division by zero.
  }
}
