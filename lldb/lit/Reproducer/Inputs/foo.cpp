struct Foo {
  Foo();
  Foo(int i);

  void A();
  void B(int i);
  int C(int i);
  int D(bool b) const;
  static void E();
  static int F(int i);
};

void Foo::A() {}
void Foo::B(int i) {}
int Foo::C(int i) { return i; }
int Foo::D(bool b) const { return 1; }
void Foo::E() {}
int Foo::F(int i) { return i; }
