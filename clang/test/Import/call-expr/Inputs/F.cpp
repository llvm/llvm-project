namespace NS {
struct X {};
void f(X) {}
void operator+(X, X) {}
} // namespace NS
void f() {
  NS::X x;
  f(x);
  x + x;
}
