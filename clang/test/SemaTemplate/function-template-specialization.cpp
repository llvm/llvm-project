// RUN: %clang_cc1 -fsyntax-only -verify %s

template <int N>
void f0(int (&array)[N]); // expected-note {{candidate template ignored: could not match 'int' against 'char'}}

// Simple function template specialization (using overloading)
template<> void f0(int (&array)[1]);

void test_f0() {
  int iarr1[1];
  f0(iarr1);
}

// Function template specialization where there are no matches
template<> void f0(char (&array)[1]); // expected-error{{no function template matches}}
template<> void f0<2>(int (&array)[2]) { }

// Function template specialization that requires partial ordering
template<typename T, int N> void f1(T (&array)[N]); // expected-note{{matches}}
template<int N> void f1(int (&array)[N]); // expected-note{{matches}}

template<> void f1(float (&array)[1]);
template<> void f1(int (&array)[1]);

// Function template specialization that results in an ambiguity
template<typename T> void f1(T (&array)[17]); // expected-note{{matches}}
template<> void f1(int (&array)[17]); // expected-error{{ambiguous}}

// Resolving that ambiguity with explicitly-specified template arguments.
template<int N> void f2(double (&array)[N]);
template<typename T> void f2(T (&array)[42]);

template<> void f2<double>(double (&array)[42]);
template<> void f2<42>(double (&array)[42]);

void f2<25>(double (&array)[25]); // expected-error{{specialization}} 

// PR5833
namespace PR5833 {
  template <typename T> bool f0(T &t1);
  template <> bool f0<float>(float &t1);
}
template <> bool PR5833::f0<float>(float &t1) {}

// PR8295
namespace PR8295 {
  template <typename T> void f(T t) {}
  template <typename T> void f<T*>(T* t) {} // expected-error{{function template partial specialization is not allowed}}
}

class Foo {
  template<class T>
  static void Bar(const T& input);

  // Don't crash here.
  template<>
  static void Bar(const long& input) {}  // expected-warning{{explicit specialization cannot have a storage class}}
};
