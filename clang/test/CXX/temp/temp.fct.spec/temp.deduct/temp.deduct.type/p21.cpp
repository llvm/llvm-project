// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

// Note: Template argument deduction involving parameter packs
// (14.5.3) can deduce zero or more arguments for each parameter pack.

template<class> struct X { 
  static const unsigned value = 0;
}; 

template<class R, class ... ArgTypes> struct X<R(int, ArgTypes ...)> { 
  static const unsigned value = 1;
}; 

template<class ... Types> struct Y { 
  static const unsigned value = 0;
}; 

template<class T, class ... Types> struct Y<T, Types& ...> { 
  static const unsigned value = 1;
};

template<class ... Types> int f(void (*)(Types ...)); 
void g(int, float);

int check0[X<int>::value == 0? 1 : -1]; // uses primary template
int check1[X<int(int, float, double)>::value == 1? 1 : -1]; // uses partial specialization
int check2[X<int(float, int)>::value == 0? 1 : -1]; // uses primary template
int check3[Y<>::value == 0? 1 : -1]; // uses primary template
int check4[Y<int&, float&, double&>::value == 1? 1 : -1]; // uses partial specialization
int check5[Y<int, float, double>::value == 0? 1 : -1]; // uses primary template
int fv = f(g); // okay
