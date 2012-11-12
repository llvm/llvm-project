// Header for PCH test blocks.c

int call_block(int (^bl)(int x, int y), int a, int b) {
  return bl(a, b);
}

int add(int a, int b) {
  return call_block(^(int x, int y) { return x + y; }, a, b);
}

int scaled_add(int a, int b, int s) {
  __block int scale = s;
  return call_block(^(int x, int y) { return x*scale + y; }, a, b);
}
