struct A {
  public:
    int x;
};

struct B : A {
  int y;
  int foo();
};

enum E {
  a = 0,
  b = 1
};
