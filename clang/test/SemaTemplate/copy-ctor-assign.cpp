// RUN: %clang_cc1 -fsyntax-only -verify %s

// Make sure that copy constructors and assignment operators are properly 
// generated when there is a matching 

// PR5072
template<typename T>
struct X {
  template<typename U> 
  X(const X<U>& other) 
    : value(other.value + 1) { } // expected-error{{binary expression}}

  template<typename U> 
  X& operator=(const X<U>& other)  {
    value = other.value + 1; // expected-error{{binary expression}}
    return *this;
  }
  
  T value;
};

struct Y {};

X<int Y::*> test0(X<int Y::*> x) { return x; }
X<int> test1(X<long> x) { return x; }


X<int> test2(X<int Y::*> x) { 
  return x; // expected-note{{instantiation}}
}

void test3(X<int> &x, X<int> xi, X<long> xl, X<int Y::*> xmptr) {
  x = xi;
  x = xl;
  x = xmptr; // expected-note{{instantiation}}
}

struct X1 {
  X1 &operator=(const X1&);
};

template<typename T>
struct X2 : X1 {
  template<typename U> X2 &operator=(const U&);
};

struct X3 : X2<int> {
};

void test_X2(X3 &to, X3 from) {
  to = from;
}
