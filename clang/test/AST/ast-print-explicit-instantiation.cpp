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

// empty <> (args deduced)
// CHECK: template void ns::foo(double);
template void ns::foo<>(double);

// CHECK: template int ns::var<int>;
template int ns::var<int>;
// CHECK: extern template float ns::var<float>;
extern template float ns::var<float>;

namespace ns {
// CHECK: template struct S<short>;
template struct S<short>;
// CHECK: template void foo<short>(short);
template void foo<short>(short);
// CHECK: template short var<short>;
template short var<short>;
}

template <typename T> struct X { struct Inner {}; };
// CHECK: template struct X<int>::Inner;
template struct X<int>::Inner;
// CHECK: extern template struct X<float>::Inner;
extern template struct X<float>::Inner;

template <typename T> struct Outer {
  void method();
  template <typename U> void f(U);
  template <typename U> static U var;
  template <typename U> struct Inner {};
  static T sval;
  static T arr[1];
};
template <typename T> void Outer<T>::method() {}
template <typename T> template <typename U> void Outer<T>::f(U) {}
template <typename T> template <typename U> U Outer<T>::var = U{};
template <typename T> T Outer<T>::sval = T{};
template <typename T> T Outer<T>::arr[1] = {};

// CHECK: template void Outer<int>::method();
template void Outer<int>::method();
// CHECK: template void Outer<int>::f<double>(double);
template void Outer<int>::f<double>(double);
// CHECK: template double Outer<int>::var<double>;
template double Outer<int>::var<double>;
// CHECK: template struct Outer<int>::Inner<double>;
template struct Outer<int>::Inner<double>;
// CHECK: template int Outer<int>::sval;
template int Outer<int>::sval;
// CHECK: template int Outer<int>::arr[1];
template int Outer<int>::arr[1];

// CHECK: extern template void Outer<float>::method();
extern template void Outer<float>::method();
// CHECK: extern template void Outer<float>::f<double>(double);
extern template void Outer<float>::f<double>(double);
// CHECK: extern template double Outer<float>::var<double>;
extern template double Outer<float>::var<double>;
// CHECK: extern template struct Outer<float>::Inner<double>;
extern template struct Outer<float>::Inner<double>;
// CHECK: extern template int Outer<float>::sval;
extern template int Outer<float>::sval;

template <typename T> struct A {
  template <typename U> struct B {
    template <typename V> void g(V);
  };
};
template <typename T> template <typename U> template <typename V>
void A<T>::B<U>::g(V) {}

// CHECK: template void A<int>::B<double>::g<float>(float);
template void A<int>::B<double>::g<float>(float);
// CHECK: extern template void A<float>::B<double>::g<int>(int);
extern template void A<float>::B<double>::g<int>(int);

namespace GH197797 {
struct S {};
enum E { X };
template <typename T> struct Wrap {};

template <typename T> T var = T{};
// CHECK: extern template S var<S>;
extern template S var<S>;
// CHECK: extern template E var<E>;
extern template E var<E>;
// CHECK: extern template Wrap<int> var<Wrap<int>>;
extern template Wrap<int> var<Wrap<int>>;

template <typename T> T var2 = T{};
// CHECK: extern template ns::S<int> var2<ns::S<int>>;
extern template ns::S<int> var2<ns::S<int>>;
// CHECK: extern template ns::S<float> var2<ns::S<float>>;
extern template ns::S<float> var2<ns::S<float>>;
} // namespace GH197797

// empty <> (default template arguments)
template <typename T = int> struct Defaulted {};
// CHECK: template struct Defaulted;
template struct Defaulted<>;

template <typename T = int> T defaulted_var = T{};
// CHECK: template int defaulted_var;
template int defaulted_var<>;
