struct A {
  virtual ~A() {}
};
struct B : public A {};

void f() {
  const A *b = new B();
  const B *c1 = dynamic_cast<const B *>(b);
  const B *c2 = static_cast<const B *>(b);
  const B *c3 = reinterpret_cast<const B *>(b);
  A *c4 = const_cast<A *>(b);
}
