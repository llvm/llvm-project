static void f() {
  int *p = 0;
  *p = 1;       // expected-warning{{Dereference of null pointer}}
}
