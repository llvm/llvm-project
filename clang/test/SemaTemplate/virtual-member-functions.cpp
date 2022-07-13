// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -triple %ms_abi_triple -DMSABI -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple %ms_abi_triple -DMSABI -fsyntax-only -std=c++98 -verify %s
// RUN: %clang_cc1 -triple %ms_abi_triple -DMSABI -fsyntax-only -std=c++11 -verify %s

namespace PR5557 {
template <class T> struct A {
  A(); // expected-note{{instantiation}}
  virtual int a(T x);
};
template<class T> A<T>::A() {}

template<class T> int A<T>::a(T x) { 
  return *x; // expected-error{{requires pointer operand}}
}

void f() {
  A<int> x; // expected-note{{instantiation}}
}

template<typename T>
struct X {
  virtual void f();
};

template<>
void X<int>::f() { }
}

// Like PR5557, but with a defined destructor instead of a defined constructor.
namespace PR5557_dtor {
template <class T> struct A {
  A(); // Don't have an implicit constructor.
  ~A(); // expected-note{{instantiation}}
  virtual int a(T x);
};
template<class T> A<T>::~A() {}

template<class T> int A<T>::a(T x) { 
  return *x; // expected-error{{requires pointer operand}}
}

void f() {
  A<int> x; // expected-note{{instantiation}}
}
}

template<typename T>
struct Base {
  virtual ~Base() { 
    int *ptr = 0;
    T t = ptr; // expected-error{{cannot initialize}}
  }
};

template<typename T>
struct Derived : Base<T> {
  virtual void foo() { }
};

template struct Derived<int>; // expected-note {{in instantiation of member function 'Base<int>::~Base' requested here}}

template<typename T>
struct HasOutOfLineKey {
  HasOutOfLineKey() { } // expected-note{{in instantiation of member function 'HasOutOfLineKey<int>::f' requested here}}
  virtual T *f(float *fp);
};

template<typename T>
T *HasOutOfLineKey<T>::f(float *fp) {
  return fp; // expected-error{{cannot initialize return object of type 'int *' with an lvalue of type 'float *'}}
}

HasOutOfLineKey<int> out_of_line; // expected-note{{in instantiation of member function 'HasOutOfLineKey<int>::HasOutOfLineKey' requested here}}

namespace std {
  class type_info;
}

namespace PR7114 {
  class A { virtual ~A(); };
#if __cplusplus <= 199711L
  // expected-note@-2{{declared private here}}
#else
  // expected-note@-4 3 {{overridden virtual function is here}}
#endif

  template<typename T>
  class B {
  public:
    class Inner : public A { };
#if __cplusplus <= 199711L
// expected-error@-2{{base class 'PR7114::A' has private destructor}}
#else
// expected-error@-4 2 {{deleted function '~Inner' cannot override a non-deleted function}}
// expected-note@-5 2 {{destructor of 'Inner' is implicitly deleted because base class 'PR7114::A' has an inaccessible destructor}}
#ifdef MSABI
// expected-note@-7 1 {{destructor of 'Inner' is implicitly deleted because base class 'PR7114::A' has an inaccessible destructor}}
#endif
#endif

    static Inner i;
    static const unsigned value = sizeof(i) == 4;
#if __cplusplus >= 201103L
// expected-note@-2 {{in instantiation of member class 'PR7114::B<int>::Inner' requested here}}
// expected-note@-3 {{in instantiation of member class 'PR7114::B<float>::Inner' requested here}}
#endif
  };

  int f() { return B<int>::value; }
#if __cplusplus >= 201103L
// expected-note@-2 {{in instantiation of template class 'PR7114::B<int>' requested here}}
#endif

#ifdef MSABI
  void test_typeid(B<float>::Inner bfi) {
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor}}
#else
// expected-error@-4 {{attempt to use a deleted function}}
// expected-note@-5 {{in instantiation of template class 'PR7114::B<float>' requested here}}
#endif

    (void)typeid(bfi);
#else
  void test_typeid(B<float>::Inner bfi) {
#if __cplusplus >= 201103L
// expected-note@-2 {{in instantiation of template class 'PR7114::B<float>' requested here}}
#endif
    (void)typeid(bfi);
#if __cplusplus <= 199711L
// expected-note@-2 {{implicit destructor}}
#endif
#endif
  }

  template<typename T>
  struct X : A {
#if __cplusplus >= 201103L
// expected-error@-2 {{deleted function '~X' cannot override a non-deleted function}}
// expected-note@-3  {{destructor of 'X<int>' is implicitly deleted because base class 'PR7114::A' has an inaccessible destructor}}
#endif
    void f() { }
  };

  void test_X(X<int> &xi, X<float> &xf) {
    xi.f();
#if __cplusplus >= 201103L
// expected-note@-2 {{in instantiation of template class 'PR7114::X<int>' requested here}}
#endif
  }
}

namespace DynamicCast {
  struct Y {};
  template<typename T> struct X : virtual Y {
    virtual void foo() { T x; }
  };
  template<typename T> struct X2 : virtual Y {
    virtual void foo() { T x; }
  };
  Y* f(X<void>* x) { return dynamic_cast<Y*>(x); }
  Y* f2(X<void>* x) { return dynamic_cast<Y*>(x); }
}

namespace avoid_using_vtable {
// We shouldn't emit the vtable for this code, in any ABI.  If we emit the
// vtable, we emit an implicit virtual dtor, which calls ~RefPtr, which requires
// a complete type for DeclaredOnly.
//
// Previously we would reference the vtable in the MS C++ ABI, even though we
// don't need to emit either the ctor or the dtor.  In the Itanium C++ ABI, the
// 'trace' method is the key function, so even though we use the vtable, we
// don't emit it.

template <typename T>
struct RefPtr {
  T *m_ptr;
  ~RefPtr() { m_ptr->deref(); }
};
struct DeclaredOnly;
struct Base {
  virtual ~Base();
};

struct AvoidVTable : Base {
  RefPtr<DeclaredOnly> m_insertionStyle;
  virtual void trace();
  AvoidVTable();
};
// Don't call the dtor, because that will emit an implicit dtor, and require a
// complete type for DeclaredOnly.
void foo() { new AvoidVTable; }
}

namespace vtable_uses_incomplete {
// Opposite of the previous test that avoids a vtable, this one tests that we
// use the vtable when the ctor is defined inline.
template <typename T>
struct RefPtr {
  T *m_ptr;
  ~RefPtr() { m_ptr->deref(); }  // expected-error {{member access into incomplete type 'vtable_uses_incomplete::DeclaredOnly'}}
};
struct DeclaredOnly; // expected-note {{forward declaration of 'vtable_uses_incomplete::DeclaredOnly'}}
struct Base {
  virtual ~Base();
};

struct UsesVTable : Base {
  RefPtr<DeclaredOnly> m_insertionStyle;
  virtual void trace();
  UsesVTable() {} // expected-note {{in instantiation of member function 'vtable_uses_incomplete::RefPtr<vtable_uses_incomplete::DeclaredOnly>::~RefPtr' requested here}}
};
}
