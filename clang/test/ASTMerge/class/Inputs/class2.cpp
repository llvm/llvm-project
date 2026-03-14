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

//Friend import tests
void f();
int g(int a);
struct X;
struct Y;

struct F1 {
public:
  int x;
  friend struct X;
  friend int g(int);
  friend void f();
};

struct F2 {
public:
  int x;
  friend struct X;
};

struct F3 {
public:
  int x;
  friend void f();
};
