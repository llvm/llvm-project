struct foo {
  int i;
};

void func(void) {
  struct foo *f;
  f->i = 3;
}
