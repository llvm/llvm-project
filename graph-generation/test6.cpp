void f();
namespace my {
void f(int);
void g();
} // namespace my
void h();

void f() {
  f();
  my::f(10);
  my::g();
}

void my::f(int n) { n = 10; }
void my::g() { h(); }

void h();

namespace you {
class B {};
} // namespace you

namespace my {
class A {
  // my::A::x(const A &, const you::B *)
  int &x(A const &a, const you::B *p);
};

int &A::x(const A &a, you::B const *p) {
  ::f();
  f(10);
}
} // namespace my

int m() {
  f();
  my::g();
  h();
}