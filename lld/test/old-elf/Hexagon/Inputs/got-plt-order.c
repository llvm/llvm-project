int c = 10;
int fn() { c = 20; return 0; }

int fn1() {
  return fn();
}
