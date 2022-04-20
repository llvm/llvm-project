// RUN: %clang_analyze_cc1 -std=c89 -analyzer-checker=core %s
x;
y(void **z) { // no-crash
  *z = x;
  int *w;
  y(&w);
  *w;
}

a;
b(*c) {}
e(*c) {
  void *d = f();
  b(d);
  *c = d;
}
void *g() {
  e(&a);
  return a;
}
j() {
  int h;
  char i = g();
  if (i)
    for (; h;)
      ;
}
