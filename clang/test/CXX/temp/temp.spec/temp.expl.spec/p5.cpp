// RUN: %clang_cc1 -fsyntax-only -verify %s

struct IntHolder {
  IntHolder(int);
};

template<typename T, typename U>
struct X {
  void f() { 
    T t;
  }
  
  void g() { }
  
  struct Inner { 
    T value; 
  };
  
  static T value;
};

template<typename T, typename U>
T X<T, U>::value;

// Explicitly specialize the members of X<IntHolder, long> to not cause
// problems with instantiation, but only provide declarations (not definitions).
template<>
void X<IntHolder, long>::f();

template<>
struct X<IntHolder, long>::Inner; // expected-note{{forward declaration}}

template<>
IntHolder X<IntHolder, long>::value;

IntHolder &test_X_IntHolderInt(X<IntHolder, long> xih) {
  xih.g(); // okay
  xih.f(); // okay, uses specialization
  
  X<IntHolder, long>::Inner inner; // expected-error {{incomplete}}
  
  return X<IntHolder, long>::value; // okay, uses specialization
}


template<class T> struct A {
  void f(T) { /* ... */ }
};

template<> struct A<int> { 
  void f(int);
};

void h() {
  A<int> a; 
  a.f(16); // A<int>::f must be defined somewhere
}

// explicit specialization syntax not used for a member of 
// explicitly specialized class template specialization 
void A<int>::f(int) { /* ... */ }
