class A {
public:
  A(int a) : a(a) {}
  int a;
};
class B : public A {
  using A::A;
};
class C : public B {
  C() : B(1) {}
};
