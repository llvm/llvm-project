// This function is broken, but this file is blacklisted
int externalBrokenFunction(int argc) {
  char x[10] = {0};
  return x[argc * 10];  // BOOM
}
