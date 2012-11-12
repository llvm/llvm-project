// RUN: %clang_cc1 -fsyntax-only -verify %s

// Tests various places where requiring a complete type involves
// instantiation of that type.

template<typename T>
struct X {
  X(T);

  T f; // expected-error{{data member instantiated with function type 'float (int)'}} \
       // expected-error{{data member instantiated with function type 'int (int)'}} \
       // expected-error{{data member instantiated with function type 'char (char)'}} \
       // expected-error{{data member instantiated with function type 'short (short)'}} \
       // expected-error{{data member instantiated with function type 'float (float)'}}
};

X<int> f() { return 0; }

struct XField {
  X<float(int)> xf; // expected-note{{in instantiation of template class 'X<float (int)>' requested here}}
};

void test_subscript(X<double> *ptr1, X<int(int)> *ptr2, int i) {
  (void)ptr1[i];
  (void)ptr2[i]; // expected-note{{in instantiation of template class 'X<int (int)>' requested here}}
}

void test_arith(X<signed char> *ptr1, X<unsigned char> *ptr2,
                X<char(char)> *ptr3, X<short(short)> *ptr4) {
  (void)(ptr1 + 5);
  (void)(5 + ptr2);
  (void)(ptr3 + 5); // expected-note{{in instantiation of template class 'X<char (char)>' requested here}}
  (void)(5 + ptr4); // expected-note{{in instantiation of template class 'X<short (short)>' requested here}}
}

void test_new() {
  (void)new X<float>(0);
  (void)new X<float(float)>; // expected-note{{in instantiation of template class 'X<float (float)>' requested here}}
}

void test_memptr(X<long> *p1, long X<long>::*pm1,
                 X<long(long)> *p2, 
                 long (X<long(long)>::*pm2)(long)) {
  (void)(p1->*pm1);
}

// Reference binding to a base
template<typename T>
struct X1 { };

template<typename T>
struct X2 : public T { };

void refbind_base(X2<X1<int> > &x2) {
  X1<int> &x1 = x2;
}

// Enumerate constructors for user-defined conversion.
template<typename T>
struct X3 {
  X3(T);
};

void enum_constructors(X1<float> &x1) {
  X3<X1<float> > x3 = x1;
}

namespace PR6376 {
  template<typename T, typename U> struct W { };

  template<typename T>
  struct X {
    template<typename U>
    struct apply {
      typedef W<T, U> type;
    };
  };
  
  template<typename T, typename U>
  struct Y : public X<T>::template apply<U>::type { };

  template struct Y<int, float>;
}

namespace TemporaryObjectCopy {
  // Make sure we instantiate classes when we create a temporary copy.
  template<typename T>
  struct X {
    X(T); 
  };

  template<typename T>
  void f(T t) {
    const X<int> &x = X<int>(t);
  }

  template void f(int);
}

namespace PR7080 {
  template <class T, class U>
  class X
  {
    typedef char true_t;
    class false_t { char dummy[2]; };
    static true_t dispatch(U);
    static false_t dispatch(...);
    static T trigger();
  public:
    enum { value = sizeof(dispatch(trigger())) == sizeof(true_t) };
  };

  template <class T>
  class rv : public T
  { };

  bool x = X<int, rv<int>&>::value;
}

namespace pr7199 {
  template <class T> class A; // expected-note {{template is declared here}}
  template <class T> class B {
    class A<T>::C field; // expected-error {{implicit instantiation of undefined template 'pr7199::A<int>'}}
  };

  template class B<int>; // expected-note {{in instantiation}}
}

namespace PR8425 {
  template <typename T>
  class BaseT {};

  template <typename T>
  class DerivedT : public BaseT<T> {};

  template <typename T>
  class FromT {
  public:
    operator DerivedT<T>() const { return DerivedT<T>(); }
  };

  void test() {
    FromT<int> ft;
    BaseT<int> bt(ft);
  }
}
