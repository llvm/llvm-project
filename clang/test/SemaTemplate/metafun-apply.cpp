// RUN: %clang_cc1 -fsyntax-only -verify %s

struct add_pointer {
  template<typename T>
  struct apply {
    typedef T* type;
  };
};

struct add_reference {
  template<typename T>
  struct apply {
    typedef T& type; // expected-error{{cannot form a reference to 'void'}}
  };
};

struct bogus {
  struct apply { // expected-note{{declared as a non-template here}}
    typedef int type;
  };
};

template<typename MetaFun, typename T>
struct apply1 {
  typedef typename MetaFun::template apply<T>::type type; // expected-note{{in instantiation of template class 'add_reference::apply<void>' requested here}} \
  // expected-error{{'apply' following the 'template' keyword does not refer to a template}}
};

int i;
apply1<add_pointer, int>::type ip = &i;
apply1<add_reference, int>::type ir = i;
apply1<add_reference, float>::type fr = i; // expected-error{{non-const lvalue reference to type 'float' cannot bind to a value of unrelated type 'int'}}

void test() {
  apply1<add_reference, void>::type t; // expected-note{{in instantiation of template class 'apply1<add_reference, void>' requested here}}

  apply1<bogus, int>::type t2; // expected-note{{in instantiation of template class 'apply1<bogus, int>' requested here}}
}


