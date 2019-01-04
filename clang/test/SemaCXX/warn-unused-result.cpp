// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

int f() __attribute__((warn_unused_result));

struct S {
  void t() const;
};
S g1() __attribute__((warn_unused_result));
S *g2() __attribute__((warn_unused_result));
S &g3() __attribute__((warn_unused_result));

void test() {
  f(); // expected-warning {{ignoring return value}}
  g1(); // expected-warning {{ignoring return value}}
  g2(); // expected-warning {{ignoring return value}}
  g3(); // expected-warning {{ignoring return value}}

  (void)f();
  (void)g1();
  (void)g2();
  (void)g3();

  if (f() == 0) return;

  g1().t();
  g2()->t();
  g3().t();

  int i = f();
  S s1 = g1();
  S *s2 = g2();
  S &s3 = g3();
  const S &s4 = g1();
}

void testSubstmts(int i) {
  switch (i) {
  case 0:
    f(); // expected-warning {{ignoring return value}}
  default:
    f(); // expected-warning {{ignoring return value}}
  }

  if (i)
    f(); // expected-warning {{ignoring return value}}
  else
    f(); // expected-warning {{ignoring return value}}

  while (i)
    f(); // expected-warning {{ignoring return value}}

  do
    f(); // expected-warning {{ignoring return value}}
  while (i);

  for (f(); // expected-warning {{ignoring return value}}
       ;
       f() // expected-warning {{ignoring return value}}
      )
    f(); // expected-warning {{ignoring return value}}

  f(),  // expected-warning {{ignoring return value}}
  (void)f();
}

struct X {
 int foo() __attribute__((warn_unused_result));
};

void bah() {
  X x, *x2;
  x.foo(); // expected-warning {{ignoring return value}}
  x2->foo(); // expected-warning {{ignoring return value}}
}

namespace warn_unused_CXX11 {
class Status;
class Foo {
 public:
  Status doStuff();
};

struct [[clang::warn_unused_result]] Status {
  bool ok() const;
  Status& operator=(const Status& x);
  inline void Update(const Status& new_status) {
    if (ok()) {
      *this = new_status; //no-warning
    }
  }
};
Status DoSomething();
Status& DoSomethingElse();
Status* DoAnotherThing();
Status** DoYetAnotherThing();
void lazy() {
  Status s = DoSomething();
  if (!s.ok()) return;
  Status &rs = DoSomethingElse();
  if (!rs.ok()) return;
  Status *ps = DoAnotherThing();
  if (!ps->ok()) return;
  Status **pps = DoYetAnotherThing();
  if (!(*pps)->ok()) return;

  (void)DoSomething();
  (void)DoSomethingElse();
  (void)DoAnotherThing();
  (void)DoYetAnotherThing();

  DoSomething(); // expected-warning {{ignoring return value}}
  DoSomethingElse();
  DoAnotherThing();
  DoYetAnotherThing();
}

template <typename T>
class [[clang::warn_unused_result]] StatusOr {
};
StatusOr<int> doit();
void test() {
  Foo f;
  f.doStuff(); // expected-warning {{ignoring return value}}
  doit(); // expected-warning {{ignoring return value}}

  auto func = []() { return Status(); };
  func(); // expected-warning {{ignoring return value}}
}
}

namespace PR17587 {
struct [[clang::warn_unused_result]] Status;

struct Foo {
  Status Bar();
};

struct Status {};

void Bar() {
  Foo f;
  f.Bar(); // expected-warning {{ignoring return value}}
};

}

namespace PR18571 {
// Unevaluated contexts should not trigger unused result warnings.
template <typename T>
auto foo(T) -> decltype(f(), bool()) { // Should not warn.
  return true;
}

void g() {
  foo(1);
}
}

namespace std {
class type_info { };
}

namespace {
// The typeid expression operand is evaluated only when the expression type is
// a glvalue of polymorphic class type.

struct B {
  virtual void f() {}
};

struct D : B {
  void f() override {}
};

struct C {};

void g() {
  // The typeid expression operand is evaluated only when the expression type is
  // a glvalue of polymorphic class type; otherwise the expression operand is not
  // evaluated and should not trigger a diagnostic.
  D d;
  C c;
  (void)typeid(f(), c); // Should not warn.
  (void)typeid(f(), d); // expected-warning {{ignoring return value}} expected-warning {{expression with side effects will be evaluated despite being used as an operand to 'typeid'}}

  // The sizeof expression operand is never evaluated.
  (void)sizeof(f(), c); // Should not warn.

   // The noexcept expression operand is never evaluated.
  (void)noexcept(f(), false); // Should not warn.
}
}

namespace {
// C++ Methods should warn even in their own class.
struct [[clang::warn_unused_result]] S {
  S DoThing() { return {}; };
  S operator++(int) { return {}; };
  S operator--(int) { return {}; };
  // Improperly written prefix.
  S operator++() { return {}; };
  S operator--() { return {}; };
};

struct [[clang::warn_unused_result]] P {
  P DoThing() { return {}; };
};

P operator++(const P &, int) { return {}; };
P operator--(const P &, int) { return {}; };
// Improperly written prefix.
P operator++(const P &) { return {}; };
P operator--(const P &) { return {}; };

void f() {
  S s;
  P p;
  s.DoThing(); // expected-warning {{ignoring return value}}
  p.DoThing(); // expected-warning {{ignoring return value}}
  // Only postfix is expected to warn when written correctly.
  s++; // expected-warning {{ignoring return value}}
  s--; // expected-warning {{ignoring return value}}
  p++; // expected-warning {{ignoring return value}}
  p--; // expected-warning {{ignoring return value}}
  // Improperly written prefix operators should still warn.
  ++s; // expected-warning {{ignoring return value}}
  --s; // expected-warning {{ignoring return value}}
  ++p; // expected-warning {{ignoring return value}}
  --p; // expected-warning {{ignoring return value}}

  // Silencing the warning by cast to void still works.
  (void)s.DoThing();
  (void)s++;
  (void)p++;
  (void)++s;
  (void)++p;
}
} // namespace

namespace PR39837 {
[[clang::warn_unused_result]] int f(int);

void g() {
  int a[2];
  for (int b : a)
    f(b); // expected-warning {{ignoring return value}}
}
} // namespace PR39837
