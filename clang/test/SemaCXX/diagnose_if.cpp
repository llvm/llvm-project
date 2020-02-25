// RUN: %clang_cc1 %s -verify -fno-builtin -std=c++14

#define _diagnose_if(...) __attribute__((diagnose_if(__VA_ARGS__)))

using size_t = decltype(sizeof(int));

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
void errorIf(T a) _diagnose_if(T() != a, "oh no", "error") {} // expected-note{{from 'diagnose_if'}}

template <typename T>
void warnIf(T a) _diagnose_if(T() != a, "oh no", "warning") {} // expected-note{{from 'diagnose_if'}}

void runIf() {
  errorIf(0);
  errorIf(1); // expected-error{{oh no}}

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
void errorIf(int a) _diagnose_if(N != a, "oh no", "error") {} // expected-note{{from 'diagnose_if'}}

template <int N>
void warnIf(int a) _diagnose_if(N != a, "oh no", "warning") {} // expected-note{{from 'diagnose_if'}}

void runIf() {
  errorIf<0>(0);
  errorIf<0>(1); // expected-error{{oh no}}

  warnIf<0>(0);
  warnIf<0>(1); // expected-warning{{oh no}}
}
}

namespace no_overload_interaction {
void foo(int) _diagnose_if(1, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
void foo(short);

void bar(int);
void bar(short) _diagnose_if(1, "oh no", "error");

void fooArg(int a) _diagnose_if(a, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
void fooArg(short);

void barArg(int);
void barArg(short a) _diagnose_if(a, "oh no", "error");

void runAll() {
  foo(1); // expected-error{{oh no}}
  bar(1);

  fooArg(1); // expected-error{{oh no}}
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

  void fooOvl(int i) _diagnose_if(i, "int bad i", "error"); // expected-note{{from 'diagnose_if'}}
  void fooOvl(short i) _diagnose_if(i, "short bad i", "error"); // expected-note{{from 'diagnose_if'}}

  void barOvl(int i) _diagnose_if(i != T(), "int bad i", "error"); // expected-note{{from 'diagnose_if'}}
  void barOvl(short i) _diagnose_if(i != T(), "short bad i", "error"); // expected-note{{from 'diagnose_if'}}
};

void runErrors() {
  Errors<int>().foo(0);
  Errors<int>().foo(1); // expected-error{{bad i}}

  Errors<int>().bar(0);
  Errors<int>().bar(1); // expected-error{{bad i}}

  Errors<int>().fooOvl(0);
  Errors<int>().fooOvl(1); // expected-error{{int bad i}}
  Errors<int>().fooOvl(short(0));
  Errors<int>().fooOvl(short(1)); // expected-error{{short bad i}}

  Errors<int>().barOvl(0);
  Errors<int>().barOvl(1); // expected-error{{int bad i}}
  Errors<int>().barOvl(short(0));
  Errors<int>().barOvl(short(1)); // expected-error{{short bad i}}
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

void bar() _diagnose_if(foo(), "bad foo", "error"); // expected-note{{from 'diagnose_if'}}
void bar(int a) _diagnose_if(foo(a), "bad foo", "error"); // expected-note{{from 'diagnose_if'}}

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
  bar(1); // expected-error{{bad foo}}
}
}

namespace late_parsed {
struct Foo {
  int i;
  constexpr Foo(int i): i(i) {}
  constexpr bool isFooable() const { return i; }

  void go() const _diagnose_if(isFooable(), "oh no", "error") {} // expected-note{{from 'diagnose_if'}}
  operator int() const _diagnose_if(isFooable(), "oh no", "error") { return 1; } // expected-note{{from 'diagnose_if'}}

  void go2() const _diagnose_if(isFooable(), "oh no", "error") // expected-note{{from 'diagnose_if'}}
      __attribute__((enable_if(true, ""))) {}
  void go2() const _diagnose_if(isFooable(), "oh no", "error") {}

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

void go(const Foo &f) _diagnose_if(f.isFooable(), "oh no", "error") {} // expected-note{{from 'diagnose_if'}}

void run() {
  Foo(0).go();
  Foo(1).go(); // expected-error{{oh no}}

  (void)int(Foo(0));
  (void)int(Foo(1)); // expected-error{{oh no}}

  Foo(0).go2();
  Foo(1).go2(); // expected-error{{oh no}}

  go(Foo(0));
  go(Foo(1)); // expected-error{{oh no}}
}
}

namespace member_templates {
struct Foo {
  int i;
  constexpr Foo(int i): i(i) {}
  constexpr bool bad() const { return i; }

  template <typename T> T getVal() _diagnose_if(bad(), "oh no", "error") { // expected-note{{from 'diagnose_if'}}
    return T();
  }

  template <typename T>
  constexpr T getVal2() const _diagnose_if(bad(), "oh no", "error") { // expected-note{{from 'diagnose_if'}}
    return T();
  }

  template <typename T>
  constexpr operator T() const _diagnose_if(bad(), "oh no", "error") { // expected-note{{from 'diagnose_if'}}
    return T();
  }

  // We hope to support emitting these errors in the future.
  int run() { return getVal<int>() + getVal2<int>() + int(*this); }
};

void run() {
  Foo(0).getVal<int>();
  Foo(1).getVal<int>(); // expected-error{{oh no}}

  Foo(0).getVal2<int>();
  Foo(1).getVal2<int>(); // expected-error{{oh no}}

  (void)int(Foo(0));
  (void)int(Foo(1)); // expected-error{{oh no}}
}
}

namespace special_member_operators {
struct Bar { int j; };
struct Foo {
  int i;
  constexpr Foo(int i): i(i) {}
  constexpr bool bad() const { return i; }
  const Bar *operator->() const _diagnose_if(bad(), "oh no", "error") { // expected-note{{from 'diagnose_if'}}
    return nullptr;
  }
  void operator()() const _diagnose_if(bad(), "oh no", "error") {} // expected-note{{from 'diagnose_if'}}
};

struct ParenOverload {
  int i;
  constexpr ParenOverload(int i): i(i) {}
  constexpr bool bad() const { return i; }
  void operator()(double) const _diagnose_if(bad(), "oh no", "error") {} // expected-note{{from 'diagnose_if'}}
  void operator()(int) const _diagnose_if(bad(), "oh no", "error") {} // expected-note{{from 'diagnose_if'}}
};

struct ParenTemplate {
  int i;
  constexpr ParenTemplate(int i): i(i) {}
  constexpr bool bad() const { return i; }
  template <typename T>
  void operator()(T) const _diagnose_if(bad(), "oh no", "error") {} // expected-note 2{{from 'diagnose_if'}}
};

void run() {
  (void)Foo(0)->j;
  (void)Foo(1)->j; // expected-error{{oh no}}

  Foo(0)();
  Foo(1)(); // expected-error{{oh no}}

  ParenOverload(0)(1);
  ParenOverload(0)(1.);

  ParenOverload(1)(1); // expected-error{{oh no}}
  ParenOverload(1)(1.); // expected-error{{oh no}}

  ParenTemplate(0)(1);
  ParenTemplate(0)(1.);

  ParenTemplate(1)(1); // expected-error{{oh no}}
  ParenTemplate(1)(1.); // expected-error{{oh no}}
}

void runLambda() {
  auto L1 = [](int i) _diagnose_if(i, "oh no", "error") {}; // expected-note{{from 'diagnose_if'}}
  L1(0);
  L1(1); // expected-error{{oh no}}
}

struct Brackets {
  int i;
  constexpr Brackets(int i): i(i) {}
  void operator[](int) _diagnose_if(i == 1, "oh no", "warning") // expected-note{{from 'diagnose_if'}}
    _diagnose_if(i == 2, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
};

void runBrackets(int i) {
  Brackets{0}[i];
  Brackets{1}[i]; // expected-warning{{oh no}}
  Brackets{2}[i]; // expected-error{{oh no}}
}

struct Unary {
  int i;
  constexpr Unary(int i): i(i) {}
  void operator+() _diagnose_if(i == 1, "oh no", "warning") // expected-note{{from 'diagnose_if'}}
    _diagnose_if(i == 2, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
};

void runUnary() {
  +Unary{0};
  +Unary{1}; // expected-warning{{oh no}}
  +Unary{2}; // expected-error{{oh no}}
}

struct PostInc {
  void operator++(int i) _diagnose_if(i == 1, "oh no", "warning") // expected-note{{from 'diagnose_if'}}
    _diagnose_if(i == 2, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
};

void runPostInc() {
  PostInc{}++;
  PostInc{}.operator++(1); // expected-warning{{oh no}}
  PostInc{}.operator++(2); // expected-error{{oh no}}
}
}

namespace ctors {
struct Foo {
  int I;
  constexpr Foo(int I): I(I) {}

  constexpr const Foo &operator=(const Foo &) const
      _diagnose_if(I, "oh no", "error") {  // expected-note{{from 'diagnose_if'}}
    return *this;
  }

  constexpr const Foo &operator=(const Foo &&) const
      _diagnose_if(I, "oh no", "error") { // expected-note{{from 'diagnose_if'}}
    return *this;
  }
};

struct Bar {
  int I;
  constexpr Bar(int I) _diagnose_if(I == 1, "oh no", "warning") // expected-note{{from 'diagnose_if'}}
    _diagnose_if(I == 2, "oh no", "error"): I(I) {} // expected-note{{from 'diagnose_if'}}
};

void run() {
  constexpr Foo F{0};
  constexpr Foo F2{1};

  F2 = F; // expected-error{{oh no}}
  F2 = Foo{2}; // expected-error{{oh no}}

  Bar{0};
  Bar{1}; // expected-warning{{oh no}}
  Bar{2}; // expected-error{{oh no}}
}
}

namespace ref_init {
struct Bar {};
struct Baz {};
struct Foo {
  int i;
  constexpr Foo(int i): i(i) {}
  operator const Bar &() const _diagnose_if(i, "oh no", "warning"); // expected-note{{from 'diagnose_if'}}
  operator const Baz &() const _diagnose_if(i, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
};
void fooBar(const Bar &b);
void fooBaz(const Baz &b);

void run() {
  fooBar(Foo{0});
  fooBar(Foo{1}); // expected-warning{{oh no}}
  fooBaz(Foo{0});
  fooBaz(Foo{1}); // expected-error{{oh no}}
}
}

namespace udl {
void operator""_fn(char c)_diagnose_if(c == 1, "oh no", "warning") // expected-note{{from 'diagnose_if'}}
    _diagnose_if(c == 2, "oh no", "error"); // expected-note{{from 'diagnose_if'}}

void run() {
  '\0'_fn;
  '\1'_fn; // expected-warning{{oh no}}
  '\2'_fn; // expected-error{{oh no}}
}
}

namespace PR31638 {
struct String {
  String(char const* __s) _diagnose_if(__s == nullptr, "oh no ptr", "warning"); // expected-note{{from 'diagnose_if'}}
  String(int __s) _diagnose_if(__s != 0, "oh no int", "warning"); // expected-note{{from 'diagnose_if'}}
};

void run() {
  String s(nullptr); // expected-warning{{oh no ptr}}
  String ss(42); // expected-warning{{oh no int}}
}
}

namespace PR31639 {
struct Foo {
  Foo(int I) __attribute__((diagnose_if(I, "oh no", "error"))); // expected-note{{from 'diagnose_if'}}
};

void bar() { Foo f(1); } // expected-error{{oh no}}
}

namespace user_defined_conversion {
struct Foo {
  int i;
  constexpr Foo(int i): i(i) {}
  operator size_t() const _diagnose_if(i == 1, "oh no", "warning") // expected-note{{from 'diagnose_if'}}
      _diagnose_if(i == 2, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
};

void run() {
  // `new T[N]`, where N is implicitly convertible to size_t, calls
  // PerformImplicitConversion directly. This lets us test the diagnostic logic
  // in PerformImplicitConversion.
  new int[Foo{0}];
  new int[Foo{1}]; // expected-warning{{oh no}}
  new int[Foo{2}]; // expected-error{{oh no}}
}
}

namespace std {
  template <typename T>
  struct initializer_list {
    const T *ptr;
    size_t elems;

    constexpr size_t size() const { return elems; }
  };
}

namespace initializer_lists {
struct Foo {
  Foo(std::initializer_list<int> l)
    _diagnose_if(l.size() == 1, "oh no", "warning") // expected-note{{from 'diagnose_if'}}
    _diagnose_if(l.size() == 2, "oh no", "error") {} // expected-note{{from 'diagnose_if'}}
};

void run() {
  Foo{std::initializer_list<int>{}};
  Foo{std::initializer_list<int>{1}}; // expected-warning{{oh no}}
  Foo{std::initializer_list<int>{1, 2}}; // expected-error{{oh no}}
  Foo{std::initializer_list<int>{1, 2, 3}};
}
}

namespace range_for_loop {
  namespace adl {
    struct Foo {
      int i;
      constexpr Foo(int i): i(i) {}
    };
    void **begin(const Foo &f) _diagnose_if(f.i, "oh no", "warning");
    void **end(const Foo &f) _diagnose_if(f.i, "oh no", "warning");

    struct Bar {
      int i;
      constexpr Bar(int i): i(i) {}
    };
    void **begin(const Bar &b) _diagnose_if(b.i, "oh no", "error");
    void **end(const Bar &b) _diagnose_if(b.i, "oh no", "error");
  }

  void run() {
    for (void *p : adl::Foo(0)) {}
    // FIXME: This should emit diagnostics. It seems that our constexpr
    // evaluator isn't able to evaluate `adl::Foo(1)` as a constant, though.
    for (void *p : adl::Foo(1)) {}

    for (void *p : adl::Bar(0)) {}
    // FIXME: Same thing.
    for (void *p : adl::Bar(1)) {}
  }
}

namespace operator_new {
struct Foo {
  int j;
  static void *operator new(size_t i) _diagnose_if(i, "oh no", "warning"); // expected-note{{from 'diagnose_if'}}
};

struct Bar {
  int j;
  static void *operator new(size_t i) _diagnose_if(!i, "oh no", "warning");
};

void run() {
  new Foo(); // expected-warning{{oh no}}
  new Bar();
}
}

namespace contextual_implicit_conv {
struct Foo {
  int i;
  constexpr Foo(int i): i(i) {}
  constexpr operator int() const _diagnose_if(i == 1, "oh no", "warning") // expected-note{{from 'diagnose_if'}}
      _diagnose_if(i == 2, "oh no", "error") { // expected-note{{from 'diagnose_if'}}
    return i;
  }
};

void run() {
  switch (constexpr Foo i = 0) { default: break; }
  switch (constexpr Foo i = 1) { default: break; } // expected-warning{{oh no}}
  switch (constexpr Foo i = 2) { default: break; } // expected-error{{oh no}}
}
}
