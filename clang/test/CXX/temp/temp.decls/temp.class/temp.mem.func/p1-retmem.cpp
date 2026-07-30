// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T> struct X1 { };

template<typename T>
struct X0 {
  typedef int size_type;
  typedef T value_type;
  
  size_type f0() const;
  value_type *f1();
  X1<value_type*> f2();
};

template<typename T>
typename X0<T>::size_type X0<T>::f0() const { 
  return 0;
}

template<typename U>
typename X0<U>::value_type *X0<U>::f1() { 
  return 0;
};

template<typename U>
X1<typename X0<U>::value_type*> X0<U>::f2() { 
  return 0;
};
