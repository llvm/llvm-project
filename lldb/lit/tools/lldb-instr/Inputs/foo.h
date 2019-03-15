struct Bar {};
struct Foo {
  Foo();
  Foo(int i);

  void A();
  void B(int i);
  int C(int i);
  int D(bool b) const;
  static void E();
  static int F(int i);
  void G(const char *fmt...);
  static Foo H();
  void I() const;
  Bar J() const;
  Bar K(void *v) const;
};
