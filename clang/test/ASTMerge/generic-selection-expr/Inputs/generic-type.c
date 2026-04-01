void f(void) {
  _Static_assert(_Generic(int, float : 0, int : 1), "Incorrect semantics of _Generic");
  _Static_assert(_Generic(float, float : 1, int : 0), "Incorrect semantics of _Generic");
}
