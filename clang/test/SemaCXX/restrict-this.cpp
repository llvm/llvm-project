// RUN: %clang_cc1 -verify -fsyntax-only -DMSVC=false %s
// RUN: %clang_cc1 -verify -fsyntax-only -fms-compatibility -DMSVC=true %s
// expected-no-diagnostics

// Check that '__restrict' is applied to 'this' as appropriate; note that a
// mismatch in '__restrict'-qualification is allowed between the declaration
// and definition of a member function. In MSVC mode, 'this' is '__restrict'
// if the *declaration* is '__restrict'; otherwise, 'this' is '__restrict' if
// the *definition* is '__restrict'.

#define Restrict(C) static_assert(__is_same(decltype(this), C* __restrict))
#define NotRestrict(C) static_assert(__is_same(decltype(this), C*))

#if MSVC
# define RestrictIfMSVC Restrict
# define RestrictIfGCC NotRestrict
#else
# define RestrictIfMSVC NotRestrict
# define RestrictIfGCC Restrict
#endif

struct C {
  void f() __restrict {
    Restrict(C);
    (void) [this]() {
      Restrict(C);
      (void) [this]() { Restrict(C); };

      // By-value capture means 'this' is now a different object; do not
      // make it __restrict.
      (void) [*this]() { RestrictIfMSVC(const C); };
      (void) [*this]() mutable { RestrictIfMSVC(C); };
    };
  }

  void a() __restrict;
  void b() __restrict;
  void c();

  template <typename U>
  void ta() __restrict;

  template <typename U>
  void tb() __restrict;

  template <typename U>
  void tc();
};

template <typename T> struct TC {
  void f() __restrict {
    Restrict(TC);
    (void) [this]() {
      Restrict(TC);
      (void) [this]() { Restrict(TC); };

      // By-value capture means 'this' is now a different object; do not
      // make it __restrict.
      (void) [*this]() { RestrictIfMSVC(const TC); };
      (void) [*this]() mutable { RestrictIfMSVC(TC); };
    };
  }

  void a() __restrict;
  void b() __restrict;
  void c();

  template <typename U>
  void ta() __restrict;

  template <typename U>
  void tb() __restrict;

  template <typename U>
  void tc();
};

// =========

void C::a() __restrict {
  Restrict(C);
  (void) [this]() {
    Restrict(C);
    (void) [*this]() { RestrictIfMSVC(const C); };
    (void) [*this]() mutable { RestrictIfMSVC(C); };
  };
}

template <typename T>
void TC<T>::a() __restrict {
  Restrict(TC);
  (void) [this]() {
    Restrict(TC);
    (void) [*this]() { RestrictIfMSVC(const TC); };
    (void) [*this]() mutable { RestrictIfMSVC(TC); };
  };
}

template <typename T>
void C::ta() __restrict {
  Restrict(C);
  (void) [this]() {
    Restrict(C);
    (void) [*this]() { RestrictIfMSVC(const C); };
    (void) [*this]() mutable { RestrictIfMSVC(C); };
  };
}

template <typename T>
template <typename U>
void TC<T>::ta() __restrict {
  Restrict(TC);
  (void) [this]() {
    Restrict(TC);
    (void) [*this]() { RestrictIfMSVC(const TC); };
    (void) [*this]() mutable { RestrictIfMSVC(TC); };
  };
}

// =========

void C::b() {
  RestrictIfMSVC(C);
  (void) [this]() {
    RestrictIfMSVC(C);
    (void) [*this]() { RestrictIfMSVC(const C); };
    (void) [*this]() mutable { RestrictIfMSVC(C); };
  };
}

template <typename T>
void TC<T>::b() {
  RestrictIfMSVC(TC);
  (void) [this]() {
    RestrictIfMSVC(TC);
    (void) [*this]() { RestrictIfMSVC(const TC); };
    (void) [*this]() mutable { RestrictIfMSVC(TC); };
  };
}

template <typename T>
void C::tb() {
  RestrictIfMSVC(C);
  (void) [this]() {
    RestrictIfMSVC(C);
    (void) [*this]() { RestrictIfMSVC(const C); };
    (void) [*this]() mutable { RestrictIfMSVC(C); };
  };
}

template <typename T>
template <typename U>
void TC<T>::tb() {
  RestrictIfMSVC(TC);
  (void) [this]() {
    RestrictIfMSVC(TC);
    (void) [*this]() { RestrictIfMSVC(const TC); };
    (void) [*this]() mutable { RestrictIfMSVC(TC); };
  };
}

// =========

void C::c() __restrict {
  RestrictIfGCC(C);
  (void) [this]() {
    RestrictIfGCC(C);
    (void) [*this]() { NotRestrict(const C); };
    (void) [*this]() mutable { NotRestrict(C); };
  };
}

template <typename T>
void TC<T>::c() __restrict {
  RestrictIfGCC(TC);
  (void) [this]() {
    RestrictIfGCC(TC);
    (void) [*this]() { NotRestrict(const TC); };
    (void) [*this]() mutable { NotRestrict(TC); };
  };
}

template <typename T>
void C::tc() __restrict {
  RestrictIfGCC(C);
  (void) [this]() {
    RestrictIfGCC(C);
    (void) [*this]() { NotRestrict(const C); };
    (void) [*this]() mutable { NotRestrict(C); };
  };
}

template <typename T>
template <typename U>
void TC<T>::tc() __restrict {
  RestrictIfGCC(TC);
  (void) [this]() {
    RestrictIfGCC(TC);
    (void) [*this]() { NotRestrict(const TC); };
    (void) [*this]() mutable { NotRestrict(TC); };
  };
}

// =========

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
