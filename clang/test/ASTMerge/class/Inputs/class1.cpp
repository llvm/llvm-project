struct A {
  public:
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

enum E {
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
  friend void f();
};

struct F3 {
public:
  int x;
  friend int g(int);
  friend void f();
};
