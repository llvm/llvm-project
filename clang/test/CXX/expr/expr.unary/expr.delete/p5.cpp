// RUN: %clang_cc1 -verify %s

// If the object being deleted has incomplete class type at the point of
// deletion and the complete class has a non-trivial destructor or a
// deallocation function, the behavior is undefined.

// The trivial case.
class T0; // expected-note {{forward declaration}}
void f0(T0 *a) { delete a; } // expected-warning {{deleting pointer to incomplete type}}
class T0 { ~T0(); };

// The trivial case, inside a template instantiation.
template<typename T>
struct T1_A { T *x; ~T1_A() { delete x; } }; // expected-warning {{deleting pointer to incomplete type}}
class T1_B; // expected-note {{forward declaration}}
void f0() { T1_A<T1_B> x; } // expected-note {{in instantiation of member function}}

// This case depends on when we check T2_C::f0.
class T2_A;
template<typename T>
struct T2_B { void f0(T *a) { delete a; } };
struct T2_C { T2_B<T2_A> x; void f0(T2_A *a) { x.f0(a); } };
void f0(T2_A *a) { T2_C x; x.f0(a); }
class T2_A { };

// An alternate version of the same.
class T3_A;
template<typename T>
struct T3_B { 
  void f0(T *a) { 
    delete a; // expected-error{{calling a private destructor of class 'T3_A'}}
  } 
};

struct T3_C { 
  T3_B<T3_A> x; 
  void f0(T3_A *a) { 
    x.f0(a); // expected-note{{in instantiation of member function 'T3_B<T3_A>::f0' requested here}}
  } 
};

void f0(T3_A *a) { T3_C x; x.f0(a); }
class T3_A { 
private: 
  ~T3_A(); // expected-note{{declared private here}}
};
