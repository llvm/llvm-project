// RUN: %clang_cc1 -fsyntax-only -pedantic-errors -verify %s
template<class T> struct A { 
  void f(T);
  template<class X1> void g1(T, X1); 
  template<class X2> void g2(T, X2); 
  void h(T) { }
};

// specialization 
template<> void A<int>::f(int);

// out of class member template definition 
template<class T> template<class X1> void A<T>::g1(T, X1) { }

// member template specialization 
template<> template<class X1> void A<int>::g1(int, X1);

// member template specialization 
template<> template<>
  void A<int>::g1(int, char);	// X1 deduced as char 

template<> template<>
  void A<int>::g2<char>(int, char); // X2 specified as char 
                                    // member specialization even if defined in class definition

template<> void A<int>::h(int) { }

namespace PR10024 {
  template <typename T> 
  struct Test{ 
    template <typename U> 
    void get(U i) {}
  }; 

  template <typename T>
  template <>
  void Test<T>::get<double>(double i) {}  // expected-error{{cannot specialize (with 'template<>') a member of an unspecialized template}}
}

namespace extraneous {
  template<typename T> struct A;

  template<typename T> int x;

  template<typename T> void f();

  template<> // expected-error{{extraneous template parameter list in template specialization}}
  template<>
  struct A<int>;

  template<> // expected-error{{extraneous template parameter list in template specialization}}
  template<>
  int x<int>;

  template<> // expected-error{{extraneous template parameter list in template specialization}}
  template<>
  void f<int>();

  template<typename T>
  struct B {
    struct C;

    template<typename U>
    struct D;

    static int y;

    template<typename U>
    static int z;

    void g();

    template<typename U>
    void h();

    enum class E;

    enum F : int;
  };

  template<>
  template<> // expected-error{{extraneous 'template<>' in declaration of struct 'C'}}
  struct B<int>::C;

  template<>
  template<> // expected-error{{extraneous template parameter list in template specialization}}
  template<>
  struct B<int>::D<int>;

  template<>
  template<> // expected-error{{extraneous template parameter list in template specialization}}
  template<typename U>
  struct B<int>::D;

  template<>
  template<> // expected-error{{extraneous 'template<>' in declaration of variable 'y'}}
  int B<int>::y;

  template<>
  template<> // expected-error{{extraneous template parameter list in template specialization}}
  template<>
  int B<int>::z<int>;

  template<>
  template<> // expected-error{{extraneous template parameter list in template specialization}}
  template<typename U>
  int B<int>::z;

  template<>
  template<>
  void B<int>::g(); // expected-error{{no function template matches function template specialization 'g'}}

  template<>
  template<> // expected-error{{extraneous template parameter list in template specialization}}
  template<>
  void B<int>::h<int>();

  template<>
  template<> // expected-error{{extraneous template parameter list in template specialization}}
  template<typename U>
  void B<int>::h<int>(); // expected-error{{function template partial specialization is not allowed}}

  // FIXME: We should diagnose this as having an extraneous 'template<>'
  template<>
  template<>
  enum class B<int>::E; // expected-error{{enumeration cannot be a template}}

  // FIXME: We should diagnose this as having an extraneous 'template<>'
  template<>
  template<>
  enum B<int>::F : int; // expected-error{{enumeration cannot be a template}}
}
