// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-return-braced-init-list %t

namespace std {
typedef decltype(sizeof(int)) size_t;

// libc++'s implementation
template <class _E>
class initializer_list {
  const _E *__begin_;
  size_t __size_;

  initializer_list(const _E *__b, size_t __s)
      : __begin_(__b),
        __size_(__s) {}

public:
  typedef _E value_type;
  typedef const _E &reference;
  typedef const _E &const_reference;
  typedef size_t size_type;

  typedef const _E *iterator;
  typedef const _E *const_iterator;

  initializer_list() : __begin_(nullptr), __size_(0) {}

  size_t size() const { return __size_; }
  const _E *begin() const { return __begin_; }
  const _E *end() const { return __begin_ + __size_; }
};

template <typename T>
class vector {
public:
  vector(T) {}
  vector(std::initializer_list<T>) {}
};
}

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

auto f3() {
  Bar b3;
  return Foo(b3);
}

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

std::vector<int> f7() {
  int i7 = 1;
  return std::vector<int>(i7);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: avoid repeating the return type
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
