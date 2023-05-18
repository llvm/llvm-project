// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace N { 
  template<class T> class X { /* ... */ }; 
  template<class T> class Y { /* ... */ };
  template<> class X<int> { /* ... */ }; 
  template<> class Y<double>;
  
  const unsigned NumElements = 17;
} 

template<> class N::Y<double> { 
  int array[NumElements];
};
