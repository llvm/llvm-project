template <typename T>
void f() {
  T x;
  _Static_assert(_Generic(x, float : 0, int : 1), "Incorrect semantics of _Generic");
}
