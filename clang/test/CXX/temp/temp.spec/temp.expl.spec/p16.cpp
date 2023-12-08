// RUN: %clang_cc1 -fsyntax-only -verify %s
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
