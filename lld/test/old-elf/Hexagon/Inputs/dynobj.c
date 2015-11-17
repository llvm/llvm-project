extern int shankar;
static int a;
static int b;
int c;
int fn2() {
  return 0;
}

int fn1() {
  return 0;
}

int fn() {
  a = 10;
  b = 20;
  c = 10;
  shankar = 20;
  return 0;
}

int fn3() {
  fn();
  fn1();
  fn2();
  return 0;
}
