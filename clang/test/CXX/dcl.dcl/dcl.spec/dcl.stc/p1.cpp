// RUN: %clang_cc1 -fsyntax-only -verify=expected,spec %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wno-explicit-specialization-storage-class %s

// A storage-class-specifier shall not be specified in an explicit
// specialization (14.7.3) or an explicit instantiation (14.7.2)
// directive.
template<typename T> void f(T) {}
template<typename T> static void g(T) {}


template<> static void f<int>(int); // spec-warning{{explicit specialization cannot have a storage class}}
template static void f<float>(float); // expected-error{{explicit instantiation cannot have a storage class}}

template<> void f<double>(double);
template void f<long>(long);

template<> static void g<int>(int); // spec-warning{{explicit specialization cannot have a storage class}}
template static void g<float>(float); // expected-error{{explicit instantiation cannot have a storage class}}

template<> void g<double>(double);
template void g<long>(long);

template<typename T>
struct X {
  static int value;
};

template<typename T>
int X<T>::value = 17;

template static int X<int>::value; // expected-error{{explicit instantiation cannot have a storage class}}

template<> static int X<float>::value; // spec-warning{{explicit specialization cannot have a storage class}}
                                       // expected-error@-1{{'static' can only be specified inside the class definition}}

struct t1 {
  template<typename>
  static void f1();
  template<>
  static void f1<int>(); // spec-warning{{explicit specialization cannot have a storage class}}
};
