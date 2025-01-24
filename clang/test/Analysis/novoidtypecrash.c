// RUN: %clang_analyze_cc1 -Wno-error=return-type -std=c89 -Wno-int-conversion -analyzer-checker=core %s
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
