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

  DoSomething(); // expected-warning {{ignoring return value of type 'Status' declared with 'clang::warn_unused_result'}}
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
  f.doStuff(); // expected-warning {{ignoring return value of type 'Status' declared with 'clang::warn_unused_result'}}
  doit(); // expected-warning {{ignoring return value of type 'StatusOr<int>' declared with 'clang::warn_unused_result'}}

  auto func = []() { return Status(); };
  func(); // expected-warning {{ignoring return value of type 'Status' declared with 'clang::warn_unused_result'}}
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
  f.Bar(); // expected-warning {{ignoring return value of type 'Status' declared with 'clang::warn_unused_result'}}
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
  s.DoThing(); // expected-warning {{ignoring return value of type 'S' declared with 'clang::warn_unused_result'}}
  p.DoThing(); // expected-warning {{ignoring return value of type 'P' declared with 'clang::warn_unused_result'}}
  // Only postfix is expected to warn when written correctly.
  s++; // expected-warning {{ignoring return value of type 'S' declared with 'clang::warn_unused_result'}}
  s--; // expected-warning {{ignoring return value of type 'S' declared with 'clang::warn_unused_result'}}
  p++; // expected-warning {{ignoring return value of type 'P' declared with 'clang::warn_unused_result'}}
  p--; // expected-warning {{ignoring return value of type 'P' declared with 'clang::warn_unused_result'}}
  // Improperly written prefix operators should still warn.
  ++s; // expected-warning {{ignoring return value of type 'S' declared with 'clang::warn_unused_result'}}
  --s; // expected-warning {{ignoring return value of type 'S' declared with 'clang::warn_unused_result'}}
  ++p; // expected-warning {{ignoring return value of type 'P' declared with 'clang::warn_unused_result'}}
  --p; // expected-warning {{ignoring return value of type 'P' declared with 'clang::warn_unused_result'}}

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
    f(b); // expected-warning {{ignoring return value of function declared with 'clang::warn_unused_result'}}
}
} // namespace PR39837

namespace PR45520 {
[[nodiscard]] bool (*f)(); // expected-warning {{'nodiscard' attribute only applies to functions, classes, or enumerations}}
[[clang::warn_unused_result]] bool (*g)();
__attribute__((warn_unused_result)) bool (*h)();

void i([[nodiscard]] bool (*fp)()); // expected-warning {{'nodiscard' attribute only applies to functions, classes, or enumerations}}
}

namespace unused_typedef_result {
[[clang::warn_unused_result]] typedef void *a;
typedef a indirect;
a af1();
indirect indirectf1();
void af2() {
  af1(); // expected-warning {{ignoring return value of type 'a' declared with 'clang::warn_unused_result'}}
  void *(*a1)();
  a1(); // no warning
  a (*a2)();
  a2(); // expected-warning {{ignoring return value of type 'a' declared with 'clang::warn_unused_result'}}
  indirectf1(); // expected-warning {{ignoring return value of type 'a' declared with 'clang::warn_unused_result'}}
}
[[nodiscard]] typedef void *b1; // expected-warning {{'[[nodiscard]]' attribute ignored when applied to a typedef; consider using '__attribute__((warn_unused_result))' or '[[clang::warn_unused_result]]' instead}}
[[gnu::warn_unused_result]] typedef void *b2; // expected-warning {{'[[gnu::warn_unused_result]]' attribute ignored when applied to a typedef; consider using '__attribute__((warn_unused_result))' or '[[clang::warn_unused_result]]' instead}}
b1 b1f1();
b2 b2f1();
void bf2() {
  b1f1(); // no warning
  b2f1(); // no warning
}
__attribute__((warn_unused_result)) typedef void *c;
c cf1();
void cf2() {
  cf1(); // expected-warning {{ignoring return value of type 'c' declared with 'warn_unused_result'}}
  void *(*c1)();
  c1();
  c (*c2)();
  c2(); // expected-warning {{ignoring return value of type 'c' declared with 'warn_unused_result'}}
}
}

namespace nodiscard_specialization {
// Test to only mark a specialization of class template as nodiscard
template<typename T> struct S { S(int) {} };
template<> struct [[nodiscard]] S<int> { S(int) {} };
template<typename T> struct [[clang::warn_unused_result]] S<const T> { S(int) {} };

template<typename T>
S<T> obtain(const T&) { return {2}; }

template<typename T>
[[nodiscard]] S<T> obtain2(const T&) { return {2}; }

template<typename T>
__attribute__((warn_unused_result)) S<T> obtain3(const T&) { return {2}; }

void use() {
  obtain(1.0);             // no warning
  obtain(1);               // expected-warning {{ignoring return value of type 'S<int>' declared with 'nodiscard'}}
  obtain<const double>(1); // expected-warning {{ignoring return value of type 'S<const double>' declared with 'clang::warn_unused_result'}}

  S<double>(2);     // no warning
  S<int>(2);        // expected-warning {{ignoring temporary of type 'S<int>' declared with 'nodiscard'}}
  S<const char>(2); // expected-warning {{ignoring temporary of type 'S<const char>' declared with 'clang::warn_unused_result' attribute}}

  // function should take precedence over type
  obtain2(1.0);             // expected-warning {{ignoring return value of function declared with 'nodiscard'}}
  obtain2(1);               // expected-warning {{ignoring return value of function declared with 'nodiscard'}}
  obtain2<const double>(1); // expected-warning {{ignoring return value of function declared with 'nodiscard'}}
  obtain3(1.0);             // expected-warning {{ignoring return value of function declared with 'warn_unused_result'}}
  obtain3(1);               // expected-warning {{ignoring return value of function declared with 'warn_unused_result'}}
  obtain3<const double>(1); // expected-warning {{ignoring return value of function declared with 'warn_unused_result'}}
}

// Test on constructor nodiscard
struct H {
  explicit H(int) {}
  [[nodiscard]] explicit H(double) {}
  __attribute__((warn_unused_result)) H(const char*) {}
};

struct [[nodiscard]] G {
  explicit G(int) {}
  [[nodiscard]] explicit G(double) {}
  [[clang::warn_unused_result]] G(const char*) {}
};

void use2() {
  H{2};       // no warning
  H(2.0);     // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard'}}
  H("Hello"); // expected-warning {{ignoring temporary created by a constructor declared with 'warn_unused_result' attribute}}

  // no warning for explicit cast to void
  (void)H(2);
  (void)H{2.0};
  (void)H{"Hello"};

  // warns for all these invocations
  // here, constructor/function should take precedence over type
  G{2};       // expected-warning {{ignoring temporary of type 'G' declared with 'nodiscard'}}
  G(2.0);     // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard'}}
  G("Hello"); // expected-warning {{ignoring temporary created by a constructor declared with 'clang::warn_unused_result'}}

  // no warning for explicit cast to void
  (void)G(2);
  (void)G{2.0};
  (void)G{"Hello"};
}
} // namespace nodiscard_specialization

namespace GH117975 {
// Test for a regression for ICE in CallExpr::getUnusedResultAttr
int f() { return 0; }
void id_print_name() {
  (int) // expected-warning {{expression result unused}}
    ((int(*)())f)();
}
} // namespace GH117975

namespace inheritance {
// Test that [[nodiscard]] is not inherited by derived class types,
// but is inherited by member functions
struct [[nodiscard]] E {
  [[nodiscard]] explicit E(int);
  explicit E(const char*);
  [[nodiscard]] int f();
};
struct F : E {
  using E::E;
};
E e();
F f();
void test() {
  e();     // expected-warning {{ignoring return value of type 'E' declared with 'nodiscard' attribute}}
  f();     // no warning: derived class type does not inherit the attribute
  E(1);    // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  E("x");  // expected-warning {{ignoring temporary of type 'E' declared with 'nodiscard' attribute}}
  F(1);    // no warning: inherited constructor does not inherit the attribute either
  F("x");  // no warning
  e().f(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  f().f(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
} // namespace inheritance

namespace BuildStringOnClangScope {

[[clang::warn_unused_result("Discarded result")]]
bool makeClangTrue() { return true; }

[[gnu::warn_unused_result("Discarded result")]]
bool makeGccTrue() { return true; }

void doClangThings() {
  makeClangTrue(); // expected-warning {{ignoring return value of function declared with 'clang::warn_unused_result' attribute: Discarded result}}
}

void doGccThings() {
  makeGccTrue(); // expected-warning {{ignoring return value of function declared with 'gnu::warn_unused_result' attribute}}
}

} // namespace BuildStringOnClangScope

namespace candiscard {

struct [[nodiscard]] NoDiscard {
  [[nodiscard]] NoDiscard(int);
  NoDiscard(const char *);
};

struct [[gnu::warn_unused]] WarnUnused {
  [[gnu::warn_unused]] WarnUnused(int); // expected-warning {{'gnu::warn_unused' attribute only applies to structs, unions, and classes}}
  WarnUnused(const char*);
};

struct [[gnu::warn_unused_result]] WarnUnusedResult {
  [[gnu::warn_unused_result]] WarnUnusedResult(int);
  WarnUnusedResult(const char*);
};

NoDiscard return_nodiscard();
WarnUnused return_warnunused();
WarnUnusedResult return_warnunusedresult();

NoDiscard (*p_return_nodiscard)();
WarnUnused (*p_return_warnunused)();
WarnUnusedResult (*p_return_warnunusedresult)();

NoDiscard (*(*pp_return_nodiscard)())();
WarnUnused (*(*pp_return_warnunused)())();
WarnUnusedResult (*(*pp_return_warnunusedresult)())();

template <class T> T from_a_template();

void test() {
  // Unused but named variables
  NoDiscard unused_variable1(1);         // no warning
  NoDiscard unused_variable2("");        // no warning
  WarnUnused unused_variable3(1);        // no warning
  WarnUnused unused_variable4("");       // no warning
  WarnUnusedResult unused_variable5(1);  // no warning
  WarnUnusedResult unused_variable6(""); // no warning

  // Constructor return values
  NoDiscard(1);         // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  NoDiscard("");        // expected-warning {{ignoring temporary of type 'NoDiscard' declared with 'nodiscard' attribute}}
  WarnUnused(1);        // expected-warning {{expression result unused}}
  WarnUnused("");       // expected-warning {{expression result unused}}
  WarnUnusedResult(1);  // expected-warning {{ignoring temporary created by a constructor declared with 'gnu::warn_unused_result' attribute}}
  WarnUnusedResult(""); // expected-warning {{ignoring temporary of type 'WarnUnusedResult' declared with 'gnu::warn_unused_result' attribute}}

  NoDiscard{1};         // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  NoDiscard{""};        // expected-warning {{ignoring temporary of type 'NoDiscard' declared with 'nodiscard' attribute}}
  WarnUnused{1};        // expected-warning {{expression result unused}}
  WarnUnused{""};       // expected-warning {{expression result unused}}
  WarnUnusedResult{1};  // expected-warning {{ignoring temporary created by a constructor declared with 'gnu::warn_unused_result' attribute}}
  WarnUnusedResult{""}; // expected-warning {{ignoring temporary of type 'WarnUnusedResult' declared with 'gnu::warn_unused_result' attribute}}

  static_cast<NoDiscard>(1);         // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  static_cast<NoDiscard>("");        // expected-warning {{ignoring temporary of type 'NoDiscard' declared with 'nodiscard' attribute}}
  static_cast<WarnUnused>(1);        // expected-warning {{expression result unused}}
  static_cast<WarnUnused>("");       // expected-warning {{expression result unused}}
  static_cast<WarnUnusedResult>(1);  // expected-warning {{ignoring temporary created by a constructor declared with 'gnu::warn_unused_result' attribute}}
  static_cast<WarnUnusedResult>(""); // expected-warning {{ignoring temporary of type 'WarnUnusedResult' declared with 'gnu::warn_unused_result' attribute}}

  // Function return values
  return_nodiscard(); // expected-warning {{ignoring return value of type 'NoDiscard' declared with 'nodiscard' attribute}}
  return_warnunused(); // no warning
  return_warnunusedresult(); // expected-warning {{ignoring return value of type 'WarnUnusedResult' declared with 'gnu::warn_unused_result' attribute}}

  // Function pointer return values
  p_return_nodiscard(); // expected-warning {{ignoring return value of type 'NoDiscard' declared with 'nodiscard' attribute}}
  p_return_warnunused(); // no warning
  p_return_warnunusedresult(); // expected-warning {{ignoring return value of type 'WarnUnusedResult' declared with 'gnu::warn_unused_result' attribute}}

  // Function pointer expression return values
  pp_return_nodiscard()(); // expected-warning {{ignoring return value of type 'NoDiscard' declared with 'nodiscard' attribute}}
  pp_return_warnunused()(); // no warning
  pp_return_warnunusedresult()(); // expected-warning {{ignoring return value of type 'WarnUnusedResult' declared with 'gnu::warn_unused_result' attribute}}

  // From a template
  from_a_template<NoDiscard>(); // expected-warning {{ignoring return value of type 'NoDiscard' declared with 'nodiscard' attribute}}
  from_a_template<WarnUnused>(); // no warning
  from_a_template<WarnUnusedResult>(); // expected-warning {{ignoring return value of type 'WarnUnusedResult' declared with 'gnu::warn_unused_result' attribute}}
}

} // namespace candiscard
