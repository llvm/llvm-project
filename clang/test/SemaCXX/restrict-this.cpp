// RUN: %clang_cc1 -verify -fsyntax-only -DMSVC=false %s
// RUN: %clang_cc1 -verify -fsyntax-only -fms-compatibility -DMSVC=true %s
// expected-no-diagnostics

// Check that '__restrict' is applied to 'this' as appropriate; note that a
// mismatch in '__restrict'-qualification is allowed between the declaration
// and definition of a member function. In MSVC mode, 'this' is '__restrict'
// if the *declaration* is '__restrict'; otherwise, 'this' is '__restrict' if
// the *definition* is '__restrict'.

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

  void a() __restrict;
  void b() __restrict;
  void c();
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

  void a() __restrict;
  void b() __restrict;
  void c();
};

// =========

void C::a() __restrict {
  static_assert(__is_same(decltype(this), C *__restrict));
  (void) [this]() {
    static_assert(__is_same(decltype(this), C *__restrict));
    (void) [this]() { static_assert(__is_same(decltype(this), C *__restrict)); };
  };
}

template <typename T>
void TC<T>::a() __restrict {
  static_assert(__is_same(decltype(this), TC<int> *__restrict));
  (void) [this]() {
    static_assert(__is_same(decltype(this), TC<int> *__restrict));
    (void) [this]() { static_assert(__is_same(decltype(this), TC<int> *__restrict)); };
  };
}

// =========

void C::b() {
  static_assert(__is_same(decltype(this), C *__restrict) == MSVC);
  (void) [this]() {
    static_assert(__is_same(decltype(this), C *__restrict) == MSVC);
    (void) [this]() { static_assert(__is_same(decltype(this), C *__restrict) == MSVC); };
  };
}

template <typename T>
void TC<T>::b() {
  static_assert(__is_same(decltype(this), TC<int> *__restrict) == MSVC);
  (void) [this]() {
    static_assert(__is_same(decltype(this), TC<int> *__restrict)  == MSVC);
    (void) [this]() { static_assert(__is_same(decltype(this), TC<int> *__restrict) == MSVC); };
  };
}

// =========

void C::c() __restrict {
  static_assert(__is_same(decltype(this), C *__restrict) == !MSVC);
  (void) [this]() {
    static_assert(__is_same(decltype(this), C *__restrict) == !MSVC);
    (void) [this]() { static_assert(__is_same(decltype(this), C *__restrict) == !MSVC); };
  };
}


template <typename T>
void TC<T>::c() __restrict {
  static_assert(__is_same(decltype(this), TC<int> *__restrict) == !MSVC);
  (void) [this]() {
    static_assert(__is_same(decltype(this), TC<int> *__restrict)  == !MSVC);
    (void) [this]() { static_assert(__is_same(decltype(this), TC<int> *__restrict) == !MSVC); };
  };
}

void f() {
  TC<int>{}.f();
  TC<int>{}.a();
  TC<int>{}.b();
  TC<int>{}.c();
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
