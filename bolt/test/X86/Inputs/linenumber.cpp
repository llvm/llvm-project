int f() {
  // This will be removed by BOLT but they make sure we have some extra space
  // to insert branches and don't run out of space when rewriting the function.
  asm("nop");
  asm("nop");
  asm("nop");
  asm("nop");
  asm("nop");
  int x = 0xBEEF;
  if (x & 0x32) {
    x++;
  } else {
    --x;
  }
  return x;
}

int g() {
  return f() + 1;
}

int main() {
  int q = g() * f();
  return 0;
}
