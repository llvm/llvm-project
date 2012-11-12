// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++98 [class.friend]p7:
// C++11 [class.friend]p9:
//   A name nominated by a friend declaration shall be accessible in
//   the scope of the class containing the friend declaration.

// PR12328
// Simple, non-templated case.
namespace test0 {
  class X {
    void f(); // expected-note {{implicitly declared private here}}
  };

  class Y {
    friend void X::f(); // expected-error {{friend function 'f' is a private member of 'test0::X'}}
  };
}

// Templated but non-dependent.
namespace test1 {
  class X {
    void f(); // expected-note {{implicitly declared private here}}
  };

  template <class T> class Y {
    friend void X::f(); // expected-error {{friend function 'f' is a private member of 'test1::X'}}
  };
}

// Dependent but instantiated at the right type.
namespace test2 {
  template <class T> class Y;

  class X {
    void f();
    friend class Y<int>;
  };

  template <class T> class Y {
    friend void X::f();
  };

  template class Y<int>;
}

// Dependent and instantiated at the wrong type.
namespace test3 {
  template <class T> class Y;

  class X {
    void f(); // expected-note {{implicitly declared private here}}
    friend class Y<int>;
  };

  template <class T> class Y {
    friend void X::f(); // expected-error {{friend function 'f' is a private member of 'test3::X'}}
  };

  template class Y<float>; // expected-note {{in instantiation}}
}

// Dependent because dependently-scoped.
namespace test4 {
  template <class T> class X {
    void f();
  };

  template <class T> class Y {
    friend void X<T>::f();
  };
}

// Dependently-scoped, no friends.
namespace test5 {
  template <class T> class X {
    void f(); // expected-note {{implicitly declared private here}}
  };

  template <class T> class Y {
    friend void X<T>::f(); // expected-error {{friend function 'f' is a private member of 'test5::X<int>'}}
  };

  template class Y<int>; // expected-note {{in instantiation}}
}

// Dependently-scoped, wrong friend.
namespace test6 {
  template <class T> class Y;

  template <class T> class X {
    void f(); // expected-note {{implicitly declared private here}}
    friend class Y<float>;
  };

  template <class T> class Y {
    friend void X<T>::f(); // expected-error {{friend function 'f' is a private member of 'test6::X<int>'}}
  };

  template class Y<int>; // expected-note {{in instantiation}}
}

// Dependently-scoped, right friend.
namespace test7 {
  template <class T> class Y;

  template <class T> class X {
    void f();
    friend class Y<int>;
  };

  template <class T> class Y {
    friend void X<T>::f();
  };

  template class Y<int>;
}
