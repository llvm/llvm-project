// RUN: %clang_cc1 %s -verify -fno-builtin -std=c++14

#define _diagnose_if(...) __attribute__((diagnose_if(__VA_ARGS__)))

namespace type_dependent {
template <typename T>
void neverok() _diagnose_if(!T(), "oh no", "error") {} // expected-note 4{{from 'diagnose_if'}}

template <typename T>
void alwaysok() _diagnose_if(T(), "oh no", "error") {}

template <typename T>
void alwayswarn() _diagnose_if(!T(), "oh no", "warning") {} // expected-note 4{{from 'diagnose_if'}}

template <typename T>
void neverwarn() _diagnose_if(T(), "oh no", "warning") {}

void runAll() {
  alwaysok<int>();
  alwaysok<int>();

  {
    void (*pok)() = alwaysok<int>;
    pok = &alwaysok<int>;
  }

  neverok<int>(); // expected-error{{oh no}}
  neverok<short>(); // expected-error{{oh no}}

  {
    void (*pok)() = neverok<int>; // expected-error{{oh no}}
  }
  {
    void (*pok)();
    pok = &neverok<int>; // expected-error{{oh no}}
  }

  alwayswarn<int>(); // expected-warning{{oh no}}
  alwayswarn<short>(); // expected-warning{{oh no}}
  {
    void (*pok)() = alwayswarn<int>; // expected-warning{{oh no}}
    pok = &alwayswarn<int>; // expected-warning{{oh no}}
  }

  neverwarn<int>();
  neverwarn<short>();
  {
    void (*pok)() = neverwarn<int>;
    pok = &neverwarn<int>;
  }
}

template <typename T>
void errorIf(T a) _diagnose_if(T() != a, "oh no", "error") {} // expected-note {{candidate disabled: oh no}}

template <typename T>
void warnIf(T a) _diagnose_if(T() != a, "oh no", "warning") {} // expected-note {{from 'diagnose_if'}}

void runIf() {
  errorIf(0);
  errorIf(1); // expected-error{{call to unavailable function}}

  warnIf(0);
  warnIf(1); // expected-warning{{oh no}}
}
}

namespace value_dependent {
template <int N>
void neverok() _diagnose_if(N == 0 || N != 0, "oh no", "error") {} // expected-note 4{{from 'diagnose_if'}}

template <int N>
void alwaysok() _diagnose_if(N == 0 && N != 0, "oh no", "error") {}

template <int N>
void alwayswarn() _diagnose_if(N == 0 || N != 0, "oh no", "warning") {} // expected-note 4{{from 'diagnose_if'}}

template <int N>
void neverwarn() _diagnose_if(N == 0 && N != 0, "oh no", "warning") {}

void runAll() {
  alwaysok<0>();
  alwaysok<1>();

  {
    void (*pok)() = alwaysok<0>;
    pok = &alwaysok<0>;
  }

  neverok<0>(); // expected-error{{oh no}}
  neverok<1>(); // expected-error{{oh no}}

  {
    void (*pok)() = neverok<0>; // expected-error{{oh no}}
  }
  {
    void (*pok)();
    pok = &neverok<0>; // expected-error{{oh no}}
  }

  alwayswarn<0>(); // expected-warning{{oh no}}
  alwayswarn<1>(); // expected-warning{{oh no}}
  {
    void (*pok)() = alwayswarn<0>; // expected-warning{{oh no}}
    pok = &alwayswarn<0>; // expected-warning{{oh no}}
  }

  neverwarn<0>();
  neverwarn<1>();
  {
    void (*pok)() = neverwarn<0>;
    pok = &neverwarn<0>;
  }
}

template <int N>
void errorIf(int a) _diagnose_if(N != a, "oh no", "error") {} // expected-note {{candidate disabled: oh no}}

template <int N>
void warnIf(int a) _diagnose_if(N != a, "oh no", "warning") {} // expected-note {{from 'diagnose_if'}}

void runIf() {
  errorIf<0>(0);
  errorIf<0>(1); // expected-error{{call to unavailable function}}

  warnIf<0>(0);
  warnIf<0>(1); // expected-warning{{oh no}}
}
}

namespace no_overload_interaction {
void foo(int) _diagnose_if(1, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
void foo(short);

void bar(int);
void bar(short) _diagnose_if(1, "oh no", "error");

void fooArg(int a) _diagnose_if(a, "oh no", "error"); // expected-note{{candidate disabled: oh no}}
void fooArg(short); // expected-note{{candidate function}}

void barArg(int);
void barArg(short a) _diagnose_if(a, "oh no", "error");

void runAll() {
  foo(1); // expected-error{{oh no}}
  bar(1);

  fooArg(1); // expected-error{{call to unavailable function}}
  barArg(1);

  auto p = foo; // expected-error{{incompatible initializer of type '<overloaded function type>'}}
}
}

namespace with_default_args {
void foo(int a = 0) _diagnose_if(a, "oh no", "warning"); // expected-note 1{{from 'diagnose_if'}}
void bar(int a = 1) _diagnose_if(a, "oh no", "warning"); // expected-note 2{{from 'diagnose_if'}}

void runAll() {
  foo();
  foo(0);
  foo(1); // expected-warning{{oh no}}

  bar(); // expected-warning{{oh no}}
  bar(0);
  bar(1); // expected-warning{{oh no}}
}
}

namespace naked_mem_expr {
struct Foo {
  void foo(int a) _diagnose_if(a, "should warn", "warning"); // expected-note{{from 'diagnose_if'}}
  void bar(int a) _diagnose_if(a, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
};

void runFoo() {
  Foo().foo(0);
  Foo().foo(1); // expected-warning{{should warn}}

  Foo().bar(0);
  Foo().bar(1); // expected-error{{oh no}}
}
}

namespace class_template {
template <typename T>
struct Errors {
  void foo(int i) _diagnose_if(i, "bad i", "error"); // expected-note{{from 'diagnose_if'}}
  void bar(int i) _diagnose_if(i != T(), "bad i", "error"); // expected-note{{from 'diagnose_if'}}

  void fooOvl(int i) _diagnose_if(i, "int bad i", "error"); // expected-note 2{{int bad i}}
  void fooOvl(short i) _diagnose_if(i, "short bad i", "error"); // expected-note 2{{short bad i}}

  void barOvl(int i) _diagnose_if(i != T(), "int bad i", "error"); // expected-note 2{{int bad i}}
  void barOvl(short i) _diagnose_if(i != T(), "short bad i", "error"); // expected-note 2{{short bad i}}
};

void runErrors() {
  Errors<int>().foo(0);
  Errors<int>().foo(1); // expected-error{{bad i}}

  Errors<int>().bar(0);
  Errors<int>().bar(1); // expected-error{{bad i}}

  Errors<int>().fooOvl(0);
  Errors<int>().fooOvl(1); // expected-error{{call to unavailable}}
  Errors<int>().fooOvl(short(0));
  Errors<int>().fooOvl(short(1)); // expected-error{{call to unavailable}}

  Errors<int>().barOvl(0);
  Errors<int>().barOvl(1); // expected-error{{call to unavailable}}
  Errors<int>().barOvl(short(0));
  Errors<int>().barOvl(short(1)); // expected-error{{call to unavailable}}
}

template <typename T>
struct Warnings {
  void foo(int i) _diagnose_if(i, "bad i", "warning"); // expected-note{{from 'diagnose_if'}}
  void bar(int i) _diagnose_if(i != T(), "bad i", "warning"); // expected-note{{from 'diagnose_if'}}

  void fooOvl(int i) _diagnose_if(i, "int bad i", "warning"); // expected-note{{from 'diagnose_if'}}
  void fooOvl(short i) _diagnose_if(i, "short bad i", "warning"); // expected-note{{from 'diagnose_if'}}

  void barOvl(int i) _diagnose_if(i != T(), "int bad i", "warning"); // expected-note{{from 'diagnose_if'}}
  void barOvl(short i) _diagnose_if(i != T(), "short bad i", "warning"); // expected-note{{from 'diagnose_if'}}
};

void runWarnings() {
  Warnings<int>().foo(0);
  Warnings<int>().foo(1); // expected-warning{{bad i}}

  Warnings<int>().bar(0);
  Warnings<int>().bar(1); // expected-warning{{bad i}}

  Warnings<int>().fooOvl(0);
  Warnings<int>().fooOvl(1); // expected-warning{{int bad i}}
  Warnings<int>().fooOvl(short(0));
  Warnings<int>().fooOvl(short(1)); // expected-warning{{short bad i}}

  Warnings<int>().barOvl(0);
  Warnings<int>().barOvl(1); // expected-warning{{int bad i}}
  Warnings<int>().barOvl(short(0));
  Warnings<int>().barOvl(short(1)); // expected-warning{{short bad i}}
}
}

namespace template_specialization {
template <typename T>
struct Foo {
  void foo() _diagnose_if(1, "override me", "error"); // expected-note{{from 'diagnose_if'}}
  void bar(int i) _diagnose_if(i, "bad i", "error"); // expected-note{{from 'diagnose_if'}}
  void baz(int i);
};

template <>
struct Foo<int> {
  void foo();
  void bar(int i);
  void baz(int i) _diagnose_if(i, "bad i", "error"); // expected-note{{from 'diagnose_if'}}
};

void runAll() {
  Foo<double>().foo(); // expected-error{{override me}}
  Foo<int>().foo();

  Foo<double>().bar(1); // expected-error{{bad i}}
  Foo<int>().bar(1);

  Foo<double>().baz(1);
  Foo<int>().baz(1); // expected-error{{bad i}}
}
}

namespace late_constexpr {
constexpr int foo();
constexpr int foo(int a);

void bar() _diagnose_if(foo(), "bad foo", "error"); // expected-note{{from 'diagnose_if'}} expected-note{{not viable: requires 0 arguments}}
void bar(int a) _diagnose_if(foo(a), "bad foo", "error"); // expected-note{{bad foo}}

void early() {
  bar();
  bar(0);
  bar(1);
}

constexpr int foo() { return 1; }
constexpr int foo(int a) { return a; }

void late() {
  bar(); // expected-error{{bad foo}}
  bar(0);
  bar(1); // expected-error{{call to unavailable function}}
}
}

namespace late_parsed {
struct Foo {
  int i;
  constexpr Foo(int i): i(i) {}
  constexpr bool isFooable() const { return i; }

  void go() const _diagnose_if(isFooable(), "oh no", "error") {} // expected-note{{from 'diagnose_if'}}
  operator int() const _diagnose_if(isFooable(), "oh no", "error") { return 1; } // expected-note{{oh no}}

  void go2() const _diagnose_if(isFooable(), "oh no", "error") // expected-note{{oh no}}
      __attribute__((enable_if(true, ""))) {}
  void go2() const _diagnose_if(isFooable(), "oh no", "error") {} // expected-note{{oh no}}

  constexpr int go3() const _diagnose_if(isFooable(), "oh no", "error")
      __attribute__((enable_if(true, ""))) {
    return 1;
  }

  constexpr int go4() const _diagnose_if(isFooable(), "oh no", "error") {
    return 1;
  }
  constexpr int go4() const _diagnose_if(isFooable(), "oh no", "error")
      __attribute__((enable_if(true, ""))) {
    return 1;
  }

  // We hope to support emitting these errors in the future. For now, though...
  constexpr int runGo() const {
    return go3() + go4();
  }
};

void go(const Foo &f) _diagnose_if(f.isFooable(), "oh no", "error") {} // expected-note{{oh no}}

void run() {
  Foo(0).go();
  Foo(1).go(); // expected-error{{oh no}}

  (void)int(Foo(0));
  (void)int(Foo(1)); // expected-error{{uses deleted function}}

  Foo(0).go2();
  Foo(1).go2(); // expected-error{{call to unavailable member function}}

  go(Foo(0));
  go(Foo(1)); // expected-error{{call to unavailable function}}
}
}

namespace member_templates {
struct Foo {
  int i;
  constexpr Foo(int i): i(i) {}
  constexpr bool bad() const { return i; }

  template <typename T> T getVal() _diagnose_if(bad(), "oh no", "error") { // expected-note{{oh no}}
    return T();
  }

  template <typename T>
  constexpr T getVal2() const _diagnose_if(bad(), "oh no", "error") { // expected-note{{oh no}}
    return T();
  }

  template <typename T>
  constexpr operator T() const _diagnose_if(bad(), "oh no", "error") { // expected-note{{oh no}}
    return T();
  }

  // We hope to support emitting these errors in the future.
  int run() { return getVal<int>() + getVal2<int>() + int(*this); }
};

void run() {
  Foo(0).getVal<int>();
  Foo(1).getVal<int>(); // expected-error{{call to unavailable member function}}

  Foo(0).getVal2<int>();
  Foo(1).getVal2<int>(); // expected-error{{call to unavailable member function}}

  (void)int(Foo(0));
  (void)int(Foo(1)); // expected-error{{uses deleted function}}
}
}

namespace special_member_operators {
struct Bar { int j; };
struct Foo {
  int i;
  constexpr Foo(int i): i(i) {}
  constexpr bool bad() const { return i; }
  const Bar *operator->() const _diagnose_if(bad(), "oh no", "error") { // expected-note{{oh no}}
    return nullptr;
  }
  void operator()() const _diagnose_if(bad(), "oh no", "error") {} // expected-note{{oh no}}
};

struct ParenOverload {
  int i;
  constexpr ParenOverload(int i): i(i) {}
  constexpr bool bad() const { return i; }
  void operator()(double) const _diagnose_if(bad(), "oh no", "error") {} // expected-note 2{{oh no}}
  void operator()(int) const _diagnose_if(bad(), "oh no", "error") {} // expected-note 2{{oh no}}
};

struct ParenTemplate {
  int i;
  constexpr ParenTemplate(int i): i(i) {}
  constexpr bool bad() const { return i; }
  template <typename T>
  void operator()(T) const _diagnose_if(bad(), "oh no", "error") {} // expected-note 2{{oh no}}
};

void run() {
  (void)Foo(0)->j;
  (void)Foo(1)->j; // expected-error{{selected unavailable operator '->'}}

  Foo(0)();
  Foo(1)(); // expected-error{{unavailable function call operator}}

  ParenOverload(0)(1);
  ParenOverload(0)(1.);

  ParenOverload(1)(1); // expected-error{{unavailable function call operator}}
  ParenOverload(1)(1.); // expected-error{{unavailable function call operator}}

  ParenTemplate(0)(1);
  ParenTemplate(0)(1.);

  ParenTemplate(1)(1); // expected-error{{unavailable function call operator}}
  ParenTemplate(1)(1.); // expected-error{{unavailable function call operator}}
}

void runLambda() {
  auto L1 = [](int i) _diagnose_if(i, "oh no", "error") {}; // expected-note{{oh no}} expected-note{{conversion candidate}}
  L1(0);
  L1(1); // expected-error{{call to unavailable function call}}
}
}

namespace ctors {
struct Foo {
  int I;
  constexpr Foo(int I): I(I) {}

  constexpr const Foo &operator=(const Foo &) const // expected-note 2{{disabled: oh no}}
      _diagnose_if(I, "oh no", "error") {
    return *this;
  }

  constexpr const Foo &operator=(const Foo &&) const // expected-note{{disabled: oh no}} expected-note{{no known conversion}}
      _diagnose_if(I, "oh no", "error") {
    return *this;
  }
};

void run() {
  constexpr Foo F{0};
  constexpr Foo F2{1};

  F2 = F; // expected-error{{selected unavailable operator}}
  F2 = Foo{2}; // expected-error{{selected unavailable operator}}
}
}
