// RUN: %check_clang_tidy -std=c++11-or-later %s modernize-return-braced-init-list %t
#include <vector>

class Bar {};

Bar b0;

class Foo {
public:
  Foo(Bar) {}
  explicit Foo(Bar, unsigned int) {}
  Foo(unsigned int) {}
};

class Baz {
public:
  Foo m() {
    Bar bm;
    return Foo(bm);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: avoid repeating the return type from the declaration; use a braced initializer list instead [modernize-return-braced-init-list]
    // CHECK-FIXES: return {bm};
  }
};

class Quux : public Foo {
public:
  Quux(Bar bar) : Foo(bar) {}
  Quux(unsigned, unsigned, unsigned k = 0) : Foo(k) {}
};

Foo f() {
  Bar b1;
  return Foo(b1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {b1};
}

Foo f2() {
  Bar b2;
  return {b2};
}

#if __cplusplus >= 201402L
auto f3() {
  Bar b3;
  return Foo(b3);
}
#endif

#define A(b) Foo(b)

Foo f4() {
  Bar b4;
  return A(b4);
}

Foo f5() {
  Bar b5;
  return Quux(b5);
}

Foo f6() {
  Bar b6;
  return Foo(b6, 1);
}

std::vector<int> vectorWithOneParameter() {
  int i7 = 1;
  return std::vector<int>(i7);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
}

std::vector<int> vectorIntWithTwoParameter() {
  return std::vector<int>(1, 2);
}

std::vector<double> vectorDoubleWithTwoParameter() {
  return std::vector<double>(1, 2.1);
}
struct A {};
std::vector<A> vectorRecordWithTwoParameter() {
  A a{};
  return std::vector<A>(1, a);
}


Bar f8() {
  return {};
}

Bar f9() {
  return Bar();
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
}

Bar f10() {
  return Bar{};
}

Foo f11(Bar b11) {
  return Foo(b11);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {b11};
}

Foo f12() {
  return f11(Bar());
}

Foo f13() {
  return Foo(Bar()); // 13
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {Bar()}; // 13
}

Foo f14() {
  // FIXME: Type narrowing should not occur!
  return Foo(-1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {-1};
}

Foo f15() {
  return Foo(f10());
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {f10()};
}

Quux f16() {
  return Quux(1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {1, 2};
}

Quux f17() {
  return Quux(1, 2, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {1, 2, 3};
}

template <typename T>
T f19() {
  return T();
}

Bar i1 = f19<Bar>();
Baz i2 = f19<Baz>();

template <typename T>
Foo f20(T t) {
  return Foo(t);
}

Foo i3 = f20(b0);

template <typename T>
class BazT {
public:
  T m() {
    Bar b;
    return T(b);
  }

  Foo m2(T t) {
    return Foo(t);
  }
};

BazT<Foo> bazFoo;
Foo i4 = bazFoo.m();
Foo i5 = bazFoo.m2(b0);

BazT<Quux> bazQuux;
Foo i6 = bazQuux.m();
Foo i7 = bazQuux.m2(b0);

auto v1 = []() { return std::vector<int>({1, 2}); }();
auto v2 = []() -> std::vector<int> { return std::vector<int>({1, 2}); }();


struct Saz {
  Saz(const int&) {}
};

Saz fn1() {
  return Saz(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {0};
}

Saz fn2() {
  int x = 1;
  return Saz(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {x};
}

struct Taz {
  Taz(const int) {}
};

Taz gn1() {
  return Taz(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {0};
}

Taz gn2() {
  int x = 0;
  return Taz(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {x};
}

Taz gn3() {
  const int& x = 0;
  return Taz(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {x};
}

struct MultiSaz {
  MultiSaz(const int&, const double) {}
};

MultiSaz mfn1() {
  int x = 1;
  double y = 2.0;
  return MultiSaz(x, y);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {x, y};
}

struct Vol {
  Vol(volatile int) {}
};

Vol vn1() {
  int x = 1;
  return Vol(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {x};
}

struct Gaz {
  Gaz(int) {}
};

Gaz hn1() {
  return Gaz(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {0};
}

Gaz hn2() {
  const int x = 1;
  return Gaz(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {x};
}

Gaz hn3() {
  const int& x = 2;
  return Gaz(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
  // CHECK-FIXES: return {x};
}
