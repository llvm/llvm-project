// RUN: %clang_cc1 -ast-print %s -o - -std=c++17 | FileCheck %s

namespace ns {
template <typename T> struct S { T x; };
template <typename T> void foo(T) {}
template <typename T> T var = T();
}

// CHECK: template struct ns::S<int>;
template struct ns::S<int>;
// CHECK: extern template struct ns::S<float>;
extern template struct ns::S<float>;

// CHECK: template void ns::foo<int>(int);
template void ns::foo<int>(int);
// CHECK: extern template void ns::foo<float>(float);
extern template void ns::foo<float>(float);

template <typename T> void bar(T) {}
// CHECK: template void bar(int);
template void bar(int);

// CHECK: template int ns::var<int>;
template int ns::var<int>;
// CHECK: extern template float ns::var<float>;
extern template float ns::var<float>;

template <typename T> struct X { struct Inner {}; };
// CHECK: template struct X<int>::Inner;
template struct X<int>::Inner;

template <typename T> struct Outer {
  void method();
  template <typename U> void f(U);
  template <typename U> static U var;
  template <typename U> struct Inner {};
};
template <typename T> void Outer<T>::method() {}
template <typename T> template <typename U> void Outer<T>::f(U) {}
template <typename T> template <typename U> U Outer<T>::var = U{};

// CHECK: template void Outer<int>::method();
template void Outer<int>::method();
// CHECK: template void Outer<int>::f<double>(double);
template void Outer<int>::f<double>(double);
// CHECK: template double Outer<int>::var<double>;
template double Outer<int>::var<double>;
// CHECK: template struct Outer<int>::Inner<double>;
template struct Outer<int>::Inner<double>;

template <typename T> struct A {
  template <typename U> struct B {
    template <typename V> void g(V);
  };
};
template <typename T> template <typename U> template <typename V>
void A<T>::B<U>::g(V) {}

// CHECK: template void A<int>::B<double>::g<float>(float);
template void A<int>::B<double>::g<float>(float);
