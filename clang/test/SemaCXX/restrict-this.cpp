// RUN: %clang_cc1 -verify -fsyntax-only %s
// expected-no-diagnostics

struct C {
  void f() __restrict {
    static_assert(__is_same(decltype(this), C *__restrict));
    (void) [this]() {
      static_assert(__is_same(decltype(this), C *__restrict));
      (void) [this]() { static_assert(__is_same(decltype(this), C *__restrict)); };

      // By-value capture means 'this' is now a different object; do not
      // make it __restrict.
      (void) [*this]() { static_assert(__is_same(decltype(this), const C *)); };
      (void) [*this]() mutable { static_assert(__is_same(decltype(this), C *)); };
    };
  }
};

template <typename T> struct TC {
  void f() __restrict {
    static_assert(__is_same(decltype(this), TC<int> *__restrict));
    (void) [this]() {
      static_assert(__is_same(decltype(this), TC<int> *__restrict));
      (void) [this]() { static_assert(__is_same(decltype(this), TC<int> *__restrict)); };

      // By-value capture means 'this' is now a different object; do not
      // make it __restrict.
      (void) [*this]() { static_assert(__is_same(decltype(this), const TC<int> *)); };
      (void) [*this]() mutable { static_assert(__is_same(decltype(this), TC<int> *)); };
    };
  }
};

void f() {
  TC<int>{}.f();
}

namespace gh18121 {
struct Foo {
  void member() __restrict {
    Foo *__restrict This = this;
  }
};
}

namespace gh42411 {
struct foo {
    int v;
    void f() const __restrict {
        static_assert(__is_same(decltype((v)), const int&));
        (void) [this]() { static_assert(__is_same(decltype((v)), const int&)); };
    }
};
}

namespace gh82941 {
void f(int& x) {
    (void)x;
}

class C {
    int x;
    void g() __restrict;
};

void C::g() __restrict {
    f(this->x);
}
}
