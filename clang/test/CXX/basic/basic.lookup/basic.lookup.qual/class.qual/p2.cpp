// RUN: %clang_cc1 -fsyntax-only -verify %s
struct X0 {
  X0 f1();
  X0 f2();
};

template<typename T>
struct X1 {
  X1<T>(int);
  (X1<T>)(float);
  X1 f2();
  X1 f2(int);
  X1 f2(float);
};

// Error recovery: out-of-line constructors whose names have template arguments.
template<typename T> X1<T>::X1<T>(int) { } // expected-error{{out-of-line constructor for 'X1' cannot have template arguments}}
template<typename T> (X1<T>::X1<T>)(float) { } // expected-error{{out-of-line constructor for 'X1' cannot have template arguments}}

// Error recovery: out-of-line constructor names intended to be types
X0::X0 X0::f1() { return X0(); } // expected-error{{qualified reference to 'X0' is a constructor name rather than a type wherever a constructor can be declared}}

struct X0::X0 X0::f2() { return X0(); }

template<typename T> X1<T>::X1<T> X1<T>::f2() { } // expected-error{{qualified reference to 'X1' is a constructor name rather than a template name wherever a constructor can be declared}}
template<typename T> X1<T>::X1<T> (X1<T>::f2)(int) { } // expected-error{{qualified reference to 'X1' is a constructor name rather than a template name wherever a constructor can be declared}}
template<typename T> struct X1<T>::X1<T> (X1<T>::f2)(float) { }
