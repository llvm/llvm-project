// XFAIL: linux

// RUN: rm -rf %t
// RUN: %clang_cc1 %s -std=c++11 -index-store-path %t/idx -DTYPE1=A -DTYPE2=A -DTYPE3=T -DTYPE4=T
// RUN: %clang_cc1 %s -std=c++11 -index-store-path %t/idx -DTYPE1=B -DTYPE2=A -DTYPE3=T -DTYPE4=T
// RUN: %clang_cc1 %s -std=c++11 -index-store-path %t/idx -DTYPE1=A -DTYPE2=B -DTYPE3=T -DTYPE4=T
// RUN: %clang_cc1 %s -std=c++11 -index-store-path %t/idx -DTYPE1=B -DTYPE2=B -DTYPE3=T -DTYPE4=T
// RUN: find %t/idx/*/records -name "record-hash*" | count 4
//
// RUN: rm -rf %t
// RUN: %clang_cc1 %s -std=c++11 -index-store-path %t/idx -DTYPE1=A -DTYPE2=A -DTYPE3=T -DTYPE4=T
// RUN: %clang_cc1 %s -std=c++11 -index-store-path %t/idx -DTYPE1=A -DTYPE2=A -DTYPE3=U -DTYPE4=T
// RUN: %clang_cc1 %s -std=c++11 -index-store-path %t/idx -DTYPE1=A -DTYPE2=A -DTYPE3=T -DTYPE4=U
// RUN: %clang_cc1 %s -std=c++11 -index-store-path %t/idx -DTYPE1=A -DTYPE2=A -DTYPE3=U -DTYPE4=U
// RUN: find %t/idx/*/records -name "record-hash*" | count 4

template<typename T>
struct A {
  typedef int X;
  void foo();
};

template<typename T>
struct B : public A<T> {
  typedef float X;
  void foo(int);
};

template<typename T>
struct C : public B<T> {
// This should result in different records, due to the different types.
  using TYPE1<T>::X;
  using TYPE2<T>::foo;
};

template <typename T>
struct D {
  typedef T X;
  void foo(T);
};
template <typename T, typename U>
struct E : public D<T>, public D<U> {
// This should result in different records, due to the different template parameter.
  using D<TYPE3>::X;
  using D<TYPE4>::foo;
};
