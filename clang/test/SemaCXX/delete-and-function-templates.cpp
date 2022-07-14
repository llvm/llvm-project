// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only  -emit-llvm-only %s
// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only  -fdelayed-template-parsing %s 
// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only  -fms-extensions %s 
// RUN: %clang_cc1 -std=c++11 -verify -fsyntax-only  -fdelayed-template-parsing -fms-extensions %s 

template<class T, class U> struct is_same { enum { value = false }; };
template<class T> struct is_same<T, T> { enum { value = true }; };

namespace test_sfinae_and_delete {

namespace ns1 {
template<class T> double f(T) = delete; //expected-note{{candidate}}
char f(...); //expected-note{{candidate}}

static_assert(is_same<decltype(f(3)),char>::value, ""); //expected-error{{call to deleted function}} expected-error{{static_assert failed}}

template<class T> decltype(f(T{})) g(T); // this one sfinae's out.
template<class T> int *g(T);
void foo() {
  int *ip = g(3);
}
} //end ns1

namespace ns2 {
template<class T> double* f(T);
template<> double* f(double) = delete;

template<class T> decltype(f(T{})) g(T); // expected-note{{candidate}}
template<class T> int *g(T); //expected-note{{candidate}}
void foo() {
  double *dp = g(3); //expected-error{{ambiguous}}
  int *ip = g(3.14); // this is OK - because the explicit specialization is deleted and sfinae's out one of the template candidates
}

} // end ns2

namespace ns3 {
template<class T> double* f(T) = delete;
template<> double* f(double);

template<class T> decltype(f(T{})) g(T); // expected-note{{candidate}}
template<class T> int *g(T); //expected-note{{candidate}}

void foo() {
  int *dp = g(3); // this is OK - because the non-double specializations are deleted and sfinae's out one of the template candidates
  double *ip = g(3.14); //expected-error{{ambiguous}}
}

} // end ns3
} // end ns test_sfinae_and_delete

namespace test_explicit_specialization_of_member {
namespace ns1 {
template<class T> struct X {
  int* f(T) = delete;
}; 
template<> int* X<int>::f(int) { }

template<class T> decltype(X<T>{}.f(T{})) g(T); // expected-note{{candidate}}
template<class T> int *g(T); //expected-note{{candidate}}

void foo() {
  int *ip2 = g(3.14); // this is OK - because the non-int specializations are deleted and sfinae's out one of the template candidates
  int *ip = g(3); //expected-error{{ambiguous}}
}

} // end ns1

namespace ns2 {
struct X {
template<class T> double* f(T) = delete;
}; 
template<> double* X::f(int);

template<class T> decltype(X{}.f(T{})) g(T); // expected-note{{candidate}}
template<class T> int *g(T); //expected-note{{candidate}}

void foo() {
  int *ip2 = g(3.14); // this is OK - because the non-int specializations are deleted and sfinae's out one of the template candidates
  int *ip = g(3); //expected-error{{ambiguous}}
}

} // end ns2

namespace ns3 {
template<class T> struct X {
  template<class U> double *f1(U, T) = delete;
  template<class U> double *f2(U, T) = delete;
};
template<> template<> double* X<int>::f1(int, int);
template<> template<class U> double* X<int>::f2(U, int);

template<class T, class U> decltype(X<T>{}.f1(U{}, T{})) g1(U, T); // expected-note{{candidate}}
template<class T, class U> int *g1(U, T); //expected-note{{candidate}}

template<class T, class U> decltype(X<T>{}.f2(U{}, T{})) g2(U, T); // expected-note2{{candidate}}
template<class T, class U> int *g2(U, T); //expected-note2{{candidate}}


void foo() {
  int *ip2 = g1(3.14, 3); // this is OK - because the non-int specializations are deleted and sfinae's out one of the template candidates
  int *ip = g1(3, 3); //expected-error{{ambiguous}}
  {
   int *ip3 = g2(3.14, 3); //expected-error{{ambiguous}}
   int *ip4 = g2(3, 3); //expected-error{{ambiguous}}
  }
  {
   int *ip3 = g2(3.14, 3.14); 
   int *ip4 = g2(3, 3.14); 
  }
}


} // end ns3

namespace ns4 {
template < typename T> T* foo (T);
template <> int* foo(int) = delete;
template <> int* foo(int); //expected-note{{candidate}}

int *IP = foo(2); //expected-error{{deleted}}
double *DP = foo(3.14);
} //end ns4

namespace ns5 {
template < typename T> T* foo (T);
template <> int* foo(int); //expected-note{{previous}}
template <> int* foo(int) = delete; //expected-error{{deleted definition must be first declaration}}

} //end ns5


} // end test_explicit_specializations_and_delete
