// RUN: %clang_cc1 %s -fsyntax-only -verify -Winfinite-recursion

void a() {  // expected-warning{{call itself}}
  a();
}

void b(int x) {  // expected-warning{{call itself}}
  if (x)
    b(x);
  else
    b(x+1);
}

void c(int x) {
  if (x)
    c(5);
}

void d(int x) {  // expected-warning{{call itself}}
  if (x)
    ++x;
  return d(x);
}

// Doesn't warn on mutually recursive functions
void e();
void f();

void e() { f(); }
void f() { e(); }

// Don't warn on infinite loops
void g() {
  while (true)
    g();

  g();
}

void h(int x) {
  while (x < 5) {
    h(x+1);
  }
}

void i(int x) {  // expected-warning{{call itself}}
  while (x < 5) {
    --x;
  }
  i(0);
}

int j() {  // expected-warning{{call itself}}
  return 5 + j();
}

class S {
  static void a();
  void b();
};

void S::a() {  // expected-warning{{call itself}}
  return a();
}

void S::b() {  // expected-warning{{call itself}}
  int i = 0;
  do {
    ++i;
    b();
  } while (i > 5);
}

template<class member>
struct T {
  member m;
  void a() { return a(); }  // expected-warning{{call itself}}
  static void b() { return b(); }  // expected-warning{{call itself}}
};

void test_T() {
  T<int> foo;
  foo.a();  // expected-note{{in instantiation}}
  foo.b();  // expected-note{{in instantiation}}
}

class U {
  U* u;
  void Fun() {  // expected-warning{{call itself}}
    u->Fun();
  }
};

// No warnings on templated functions
// sum<0>() is instantiated, does recursively call itself, but never runs.
template <int value>
int sum() {
  return value + sum<value/2>();
}

template<>
int sum<1>() { return 1; }

template<int x, int y>
int calculate_value() {
  if (x != y)
    return sum<x - y>();  // This instantiates sum<0>() even if never called.
  else
    return 0;
}

int value = calculate_value<1,1>();

void DoSomethingHere();

// DoStuff<0,0>() is instantiated, but never called.
template<int First, int Last>
int DoStuff() {
  if (First + 1 == Last) {
    // This branch gets removed during <0, 0> instantiation in so CFG for this
    // function goes straight to the else branch.
    DoSomethingHere();
  } else {
    DoStuff<First, (First + Last)/2>();
    DoStuff<(First + Last)/2, Last>();
  }
  return 0;
}
int stuff = DoStuff<0, 1>();

template<int x>
struct Wrapper {
  static int run() {
    // Similar to the above, Wrapper<0>::run() will discard the if statement.
    if (x == 1)
      return 0;
    return Wrapper<x/2>::run();
  }
  static int run2() {  // expected-warning{{call itself}}
    return run2();
  }
};

template <int x>
int test_wrapper() {
  if (x != 0)
    return Wrapper<x>::run() +
           Wrapper<x>::run2();  // expected-note{{instantiation}}
  return 0;
}

int wrapper_sum = test_wrapper<2>();  // expected-note{{instantiation}}
