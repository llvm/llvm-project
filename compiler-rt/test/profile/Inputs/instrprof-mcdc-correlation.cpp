void test(bool a, bool b, bool c, bool d) {
  if ((a && b) || (c && d))
    ;
  if (b && c)
    ;
}

int main() {
  test(true, true, true, true);
  test(true, true, false, true);
  test(true, false, true, true);
  (void)0;
  return 0;
}
