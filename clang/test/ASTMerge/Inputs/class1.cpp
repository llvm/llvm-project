struct A {
  int x;
};

struct B : A {
  float y;
  float foo();
};

struct C {
  C(int i = 10);
  C(const C&);
  C &operator=(C&);
  ~C();
};
