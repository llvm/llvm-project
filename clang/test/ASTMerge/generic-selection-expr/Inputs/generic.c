void f(void) {
  int x;
  float y;
  _Static_assert(_Generic(x, float : 0, int : 1), "Incorrect semantics of _Generic");
  _Static_assert(_Generic(y, float : 1, int : 0), "Incorrect semantics of _Generic");
}
