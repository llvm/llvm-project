int f(int x) {
  int result = (x / 42);

  if (result > 10)
    result = 10;
  else
    result = 0;

  return result;
}
