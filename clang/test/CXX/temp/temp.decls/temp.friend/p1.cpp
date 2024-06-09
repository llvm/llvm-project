// RUN: %clang_cc1 -verify -emit-llvm-only %s

namespace test0 {
template <typename T> struct Num {
  T value_;

public:
  Num(T value) : value_(value) {}
  T get() const { return value_; }

  template <typename U> struct Rep {
    U count_;
    Rep(U count) : count_(count) {}

    friend Num operator*(const Num &a, const Rep &n) {
      Num x = 0;
      for (U count = n.count_; count; --count)
        x += a;
      return x;
    }
  };

  friend Num operator+(const Num &a, const Num &b) {
    return a.value_ + b.value_;
  }

  Num& operator+=(const Num& b) {
    value_ += b.value_;
    return *this;
  }

  class Representation {};
  friend class Representation;
};

class A {
  template <typename T> friend bool iszero(const A &a) throw();
};

template <class T> class B_iterator;
template <class T> class B {
  friend class B_iterator<T>;
};

int calc1() {
  Num<int> left = -1;
  Num<int> right = 1;
  Num<int> result = left + right;
  return result.get();
}

int calc2() {
  Num<int> x = 3;
  Num<int>::Rep<char> n = (char) 10;
  Num<int> result = x * n;
  return result.get();
}
}

// Reduced from GNU <locale>
namespace test1 {
  class A {
    bool b; // expected-note {{declared private here}}
    template <typename T> friend bool has(const A&);
  };
  template <typename T> bool has(const A &x) {
    return x.b;
  }
  template <typename T> bool hasnot(const A &x) {
    return x.b; // expected-error {{'b' is a private member of 'test1::A'}}
  }
}

namespace test2 {
  class A {
    bool b; // expected-note {{declared private here}}
    template <typename T> friend class HasChecker;
  };
  template <typename T> class HasChecker {
    bool check(A *a) {
      return a->b;
    }
  };
  template <typename T> class HasNotChecker {
    bool check(A *a) {
      return a->b; // expected-error {{'b' is a private member of 'test2::A'}}
    }
  };
}

namespace test3 {
  class Bool;
  template <class T> class User;
  template <class T> T transform(class Bool, T);

  class Bool {
    friend class User<bool>;
    friend bool transform<>(Bool, bool);

    bool value; // expected-note 2 {{declared private here}}
  };

  template <class T> class User {
    static T compute(Bool b) {
      return b.value; // expected-error {{'value' is a private member of 'test3::Bool'}}
    }
  };

  template <class T> T transform(Bool b, T value) {
    if (b.value) // expected-error {{'value' is a private member of 'test3::Bool'}}
      return value;
    return value + 1;
  }

  template bool transform(Bool, bool);
  template int transform(Bool, int); // expected-note {{requested here}}

  template class User<bool>;
  template class User<int>; // expected-note {{requested here}}
}

namespace test4 {
  template <class T> class A {
    template <class T0> friend class B;
    bool foo(const A<T> *) const;
  };

  template <class T> class B {
    bool bar(const A<T> *a, const A<T> *b) {
      return a->foo(b);
    }
  };

  template class B<int>;
}

namespace test5 {
  template <class T, class U=int> class A {};
  template <class T> class B {
    template <class X, class Y> friend class A;
  };
  template class B<int>;
  template class A<int>;
}

namespace Dependent {
  template<typename T, typename Traits> class X;
  template<typename T, typename Traits>
  X<T, Traits> operator+(const X<T, Traits>&, const T*);

  template<typename T, typename Traits> class X {
    typedef typename Traits::value_type value_type;
    friend X operator+<>(const X&, const value_type*);
  };
}

namespace test7 {
  template <class T> class A { // expected-note {{declared here}}
    friend class B;
    int x; // expected-note {{declared private here}}
  };

  class B {
    int foo(A<int> &a) {
      return a.x;
    }
  };

  class C {
    int foo(A<int> &a) {
      return a.x; // expected-error {{'x' is a private member of 'test7::A<int>'}}
    }
  };

  // This shouldn't crash.
  template <class T> class D {
    friend class A; // expected-error {{template 'A' cannot be referenced with the 'class' specifier}}
  };
  template class D<int>;
}

namespace test8 {
  template <class N> class A {
    static int x;
    template <class T> friend void foo();
  };
  template class A<int>;

  template <class T> void foo() {
    A<int>::x = 0;
  }
  template void foo<int>();
}

namespace test9 {
  template <class T> class A {
    class B; class C;

    int foo(B *b) {
      return b->x;
    }

    int foo(C *c) {
      return c->x; // expected-error {{'x' is a private member}}
    }

    class B {
      int x;
      friend int A::foo(B*);
    };

    class C {
      int x; // expected-note {{declared private here}}
    };
  };

  template class A<int>; // expected-note {{in instantiation}}
}

namespace test10 {
  template <class T> class A;
  template <class T> A<T> bar(const T*, const A<T>&);
  template <class T> class A {
  private:
    void foo(); // expected-note {{declared private here}}
    friend A bar<>(const T*, const A<T>&);
  };

  template <class T> A<T> bar(const T *l, const A<T> &r) {
    A<T> l1;
    l1.foo();

    A<char> l2;
    l2.foo(); // expected-error {{'foo' is a private member of 'test10::A<char>'}}

    return l1;
  }

  template A<int> bar<int>(const int *, const A<int> &); // expected-note {{in instantiation}}
}

// PR6752: this shouldn't crash.
namespace test11 {
  struct Foo {
    template<class A>
    struct IteratorImpl {
      template<class T> friend class IteratorImpl;
    };
  };

  template struct Foo::IteratorImpl<int>;
  template struct Foo::IteratorImpl<long>;
}

// PR6827
namespace test12 {
  template <typename T> class Foo;
  template <typename T> Foo<T> foo(T* t){ return Foo<T>(t, true); }

  template <typename T> class Foo {
  public:
    Foo(T*);
    friend Foo<T> foo<T>(T*);
  private:
    Foo(T*, bool); // expected-note {{declared private here}}
  };

  // Should work.
  int globalInt;
  Foo<int> f = foo(&globalInt);

  // Shouldn't work.
  long globalLong;
  template <> Foo<long> foo(long *t) {
    Foo<int> s(&globalInt, false); // expected-error {{calling a private constructor}}
    return Foo<long>(t, true);
  }
}

// PR6514
namespace test13 {
  template <int N, template <int> class Temp>
  class Role : public Temp<N> {
    friend class Temp<N>;
    int x;
  };

  template <int N> class Foo {
    void foo(Role<N, test13::Foo> &role) {
      (void) role.x;
    }
  };

  template class Foo<0>;
}

namespace test14 {
  template <class T> class B;
  template <class T> class A {
    friend void B<T>::foo();
    static void foo(); // expected-note {{declared private here}}
  };

  template <class T> class B {
  public:
    void foo() { return A<long>::foo(); } // expected-error {{'foo' is a private member of 'test14::A<long>'}}
  };

  template class B<int>; // expected-note {{in instantiation}}
}

namespace test15 {
  template <class T> class B;
  template <class T> class A {
    friend void B<T>::foo();

    // This shouldn't be misrecognized as a templated-scoped reference.
    template <class U> friend void B<T>::bar(U);

    static void foo(); // expected-note {{declared private here}}
  };

  template <class T> class B {
  public:
    void foo() { return A<long>::foo(); } // expected-error {{'foo' is a private member of 'test15::A<long>'}}
  };

  template <> class B<float> {
  public:
    void foo() { return A<float>::foo(); }
    template <class U> void bar(U u) {
      (void) A<float>::foo();
    }
  };

  template class B<int>; // expected-note {{in instantiation}}
}

namespace PR10913 {
  template<class T> class X;

  template<class T> void f(X<T> *x) {
    x->member = 0;
  }

  template<class U, class T> void f2(X<T> *x) {
    x->member = 0; // expected-error{{'member' is a protected member of 'PR10913::X<int>'}}
  }

  template<class T> class X {
    friend void f<T>(X<T> *x);
    friend void f2<T>(X<int> *x);

  protected:
    int member; // expected-note{{declared protected here}}
  };

  template void f(X<int> *);
  template void f2<int>(X<int> *);
  template void f2<float>(X<int> *); // expected-note{{in instantiation of function template specialization 'PR10913::f2<float, int>' requested here}}
}

namespace test16 {
template <class T> struct foo {}; // expected-note{{candidate ignored: not a function template}}
template <class T> class A {
  friend void foo<T>(); // expected-error{{no candidate function template was found for dependent friend function template specialization}}
};
}

namespace test17 {
namespace ns {
template <class T> void foo() {} // expected-note{{candidate ignored: not a member of the enclosing namespace; did you mean to explicitly qualify the specialization?}}
}
using ns::foo;
template <class T> struct A {
  friend void foo<T>() {} // expected-error{{no candidate function template was found for dependent friend function template specialization}}
};
}

namespace test18 {
namespace ns1 { template <class T> struct foo {}; } // expected-note{{candidate ignored: not a function template}}
namespace ns2 { void foo() {} } // expected-note{{candidate ignored: not a function template}}
using ns1::foo; // expected-note {{found by name lookup}}
using ns2::foo; // expected-note {{found by name lookup}}

template <class T> class A {
  friend void foo<T>() {} // expected-error {{ambiguous}} expected-error{{no candidate function template was found for dependent friend function template specialization}}
};
}
