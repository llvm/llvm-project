// RUN:  %clang_cc1 -std=c++2c -verify %s

namespace ex1 {
struct C {
  union {
    int a;
    const char* p;
  };
  int x;
};

constexpr C c = { .a = 1, .x = 3 };
static_assert(c.a == 1);
static_assert(c.x == 3);

static constexpr C c2 = { .a = 1.0, .x = 3 };
// expected-error@-1 {{type 'double' cannot be narrowed to 'int' in initializer list}}
//   expected-note@-2 {{insert an explicit cast to silence this issue}}
} // namespace ex1

namespace ex2 {
struct A {
  int x;
  struct B {
    int i;
    int j;
  } b;
};

constexpr A a = { 1, { 2, 3 } };
static_assert(a.x == 1);
static_assert(a.b.i == 2);
static_assert(a.b.j == 3);

struct base1 { int b1, b2 = 42; };
struct base2 {
  constexpr base2() {
    b3 = 43;
  }
  int b3;
};
struct derived : base1, base2 {
  int d;
};

constexpr derived d1{{1, 2}, {}, 4};
static_assert(d1.b1 == 1);
static_assert(d1.b2 == 2);
static_assert(d1.b3 == 43);
static_assert(d1.d == 4);

constexpr derived d2{{}, {}, 4};
static_assert(d2.b1 == 0);
static_assert(d2.b2 == 42);
static_assert(d2.b3 == 43);
static_assert(d2.d == 4);
} // namespace ex2

namespace ex3 {
struct S {
  int a;
  const char* b;
  int c;
  int d = b[a];
};

constexpr S ss = { 1, "asdf" };
static_assert(ss.a == 1);
static_assert(__builtin_strcmp(ss.b, "asdf") == 0);
static_assert(ss.c == int{});
static_assert(ss.d == ss.b[ss.a]);

struct string {
  int d = 43;
};

struct A {
  string a;
  int b = 42;
  int c = -1;
};

constexpr A a{.c = 21};
static_assert(a.a.d == string{}.d);
static_assert(a.b == 42);
static_assert(a.c == 21);
} // namespace ex3

namespace ex4 {
int x[] = { 1, 3, 5 };
static_assert(sizeof(x) / sizeof(int) == 3);
} // namespace ex4

namespace ex5 {
struct X { int i, j, k; };

constexpr X a[] = { 1, 2, 3, 4, 5, 6 };
constexpr X b[2] = { { 1, 2, 3 }, { 4, 5, 6 } };
static_assert(sizeof(a) == sizeof(b));
static_assert(a[0].i == b[0].i);
static_assert(a[0].j == b[0].j);
static_assert(a[0].k == b[0].k);
static_assert(a[1].i == b[1].i);
static_assert(a[1].j == b[1].j);
static_assert(a[1].k == b[1].k);
} // namespace ex5

namespace ex6 {
struct S {
  int y[] = { 0 };
  // expected-error@-1 {{array bound cannot be deduced from a default member initializer}}
};
} // namespace ex6

namespace ex7 {
struct A {
  int i;
  static int s;
  int j;
  int :17;
  int k;
};

constexpr A a = { 1, 2, 3 };
static_assert(a.i == 1);
static_assert(a.j == 2);
static_assert(a.k == 3);
} // namespace ex7

namespace ex8 {
struct A;
extern A a;
struct A {
  const A& a1 { A{a,a} };
  const A& a2 { A{} };
  // expected-error@-1 {{default member initializer for 'a2' needed within definition of enclosing class 'A' outside of member functions}}
  //   expected-note@-2 {{default member initializer declared here}}
};
A a{a,a};

struct B {
  int n = B{}.n;
  // expected-error@-1 {{default member initializer for 'n' needed within definition of enclosing class 'B' outside of member functions}}
  //   expected-note@-2 {{default member initializer declared here}}
};
} // namespace ex8

namespace ex9 {
constexpr int x[2][2] = { 3, 1, 4, 2 };
static_assert(x[0][0] == 3);
static_assert(x[0][1] == 1);
static_assert(x[1][0] == 4);
static_assert(x[1][1] == 2);

constexpr float y[4][3] = {
  { 1 }, { 2 }, { 3 }, { 4 }
};
static_assert(y[0][0] == 1);
static_assert(y[0][1] == 0);
static_assert(y[0][2] == 0);
static_assert(y[1][0] == 2);
static_assert(y[1][1] == 0);
static_assert(y[1][2] == 0);
static_assert(y[2][0] == 3);
static_assert(y[2][1] == 0);
static_assert(y[2][2] == 0);
static_assert(y[3][0] == 4);
static_assert(y[3][1] == 0);
static_assert(y[3][2] == 0);
} // namespace ex9

namespace ex10 {
struct S1 { int a, b; };
struct S2 { S1 s, t; };

constexpr S2 x[2] = { 1, 2, 3, 4, 5, 6, 7, 8 };
constexpr S2 y[2] = {
  {
    { 1, 2 },
    { 3, 4 }
  },
  {
    { 5, 6 },
    { 7, 8 }
  }
};
static_assert(x[0].s.a == 1);
static_assert(x[0].s.b == 2);
static_assert(x[0].t.a == 3);
static_assert(x[0].t.b == 4);
static_assert(x[1].s.a == 5);
static_assert(x[1].s.b == 6);
static_assert(x[1].t.a == 7);
static_assert(x[1].t.b == 8);
} // namespace ex10

namespace ex11 {
char cv[4] = { 'a', 's', 'd', 'f', 0 };
// expected-error@-1 {{excess elements in array initializer}}
} // namespace ex11

namespace ex12 {
constexpr float y[4][3] = {
  { 1, 3, 5 },
  { 2, 4, 6 },
  { 3, 5, 7 },
};
static_assert(y[0][0] == 1);
static_assert(y[0][1] == 3);
static_assert(y[0][2] == 5);
static_assert(y[1][0] == 2);
static_assert(y[1][1] == 4);
static_assert(y[1][2] == 6);
static_assert(y[2][0] == 3);
static_assert(y[2][1] == 5);
static_assert(y[2][2] == 7);
static_assert(y[3][0] == 0.0);
static_assert(y[3][1] == 0.0);
static_assert(y[3][2] == 0.0);

constexpr float z[4][3] = {
  1, 3, 5, 2, 4, 6, 3, 5, 7
};
static_assert(z[0][0] == 1);
static_assert(z[0][1] == 3);
static_assert(z[0][2] == 5);
static_assert(z[1][0] == 2);
static_assert(z[1][1] == 4);
static_assert(z[1][2] == 6);
static_assert(z[2][0] == 3);
static_assert(z[2][1] == 5);
static_assert(z[2][2] == 7);
static_assert(z[3][0] == 0.0);
static_assert(z[3][1] == 0.0);
static_assert(z[3][2] == 0.0);
} // namespace ex12

namespace ex13 {
struct S { } s;
struct A {
  S s1;
  int i1;
  S s2;
  int i2;
  S s3;
  int i3;
} a = {
  { },              // Required initialization
  0,
  s,                // Required initialization
  0
};                  // Initialization not required for A​::​s3 because A​::​i3 is also not initialized
} // namespace ex13

namespace ex14 {
struct A {
  int i;
  constexpr operator int() const { return 42; };
};
struct B {
  A a1, a2;
  int z;
};
constexpr A a{};
constexpr B b = { 4, a, a };
static_assert(b.a1.i == 4);
static_assert(b.a2.i == a.i);
static_assert(b.z == a.operator int());
} // namespace ex14

namespace ex15 {
union u { // #ex15-u
  int a;
  const char* b;
};

u a = { 1 };
u b = a;
u c = 1;
// expected-error@-1 {{no viable conversion from 'int' to 'u'}}
//   expected-note@#ex15-u {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const u &' for 1st argument}}
//   expected-note@#ex15-u {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'u &&' for 1st argument}}
u d = { 0, "asdf" };
// expected-error@-1 {{excess elements in union initializer}}
u e = { "asdf" };
// expected-error@-1 {{cannot initialize a member subobject of type 'int' with an lvalue of type 'const char[5]'}}
u f = { .b = "asdf" };
u g = {
  .a = 1, // #ex15-g-a
  .b = "asdf"
  // expected-error@-1 {{initializer partially overrides prior initialization of this subobject}}
  //   expected-note@#ex15-g-a {{previous initialization is here}}
};
} // namespace ex15
