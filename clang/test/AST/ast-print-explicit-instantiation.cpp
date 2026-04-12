// RUN: %clang_cc1 -ast-print %s -o - -std=c++17 | FileCheck %s

namespace ns {
template <typename T> struct S { T x; };
template <typename T> void foo(T) {}
template <typename T> T var = T();
}

// Class template explicit instantiation.
// CHECK: template struct ns::S<int>;
template struct ns::S<int>;
// CHECK: extern template struct ns::S<float>;
extern template struct ns::S<float>;

// Function template explicit instantiation.
// CHECK: template void ns::foo<int>(int);
template void ns::foo<int>(int);
// CHECK: extern template void ns::foo<float>(float);
extern template void ns::foo<float>(float);

// Function template without explicit template args.
template <typename T> void bar(T) {}
// CHECK: template void bar(int);
template void bar(int);

// Variable template explicit instantiation.
// CHECK: template int ns::var<int>;
template int ns::var<int>;
// CHECK: extern template float ns::var<float>;
extern template float ns::var<float>;

// Nested class of class template.
template <typename T> struct X { struct Inner {}; };
// CHECK: template struct X<int>::Inner;
template struct X<int>::Inner;

// Member function of class template.
template <typename T> struct Outer { void method(); };
template <typename T> void Outer<T>::method() {}
// CHECK: template void Outer<int>::method();
template void Outer<int>::method();
