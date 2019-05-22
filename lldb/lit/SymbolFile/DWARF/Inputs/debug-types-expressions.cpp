struct A {
  int i = 47;
  int f() { return i; }
  virtual ~A() = default;
};

struct B: public A {
  int j = 42;
};

namespace ns {
struct A {
  int i = 147;
  A();
};
A::A() = default;
}

int foo(A *a) {
  return a->f();
}

int main() {
  return foo(new B);
}
