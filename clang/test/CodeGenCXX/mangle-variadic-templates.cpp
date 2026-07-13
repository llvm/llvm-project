// RUN: %clang_cc1 -Wno-error=return-type -std=c++11 -emit-llvm -triple=x86_64-apple-darwin9 -o - %s | FileCheck %s

template<unsigned I, typename ...Types>
struct X { };

template<typename T> struct identity { using type = T; };
template<typename T> struct add_reference;
template<typename ...Types> struct tuple { };
template<int ...Values> struct int_tuple { };
template<template<typename> class ...Templates> struct template_tuple { };
template<typename ...T> using ArrayOfN = int[sizeof...(T)];

// CHECK-LABEL: define weak_odr void @_Z2f0IJEEv1XIXsZT_EJDpRT_EE
template<typename ...Types>
void f0(X<sizeof...(Types), Types&...>) { }

template void f0(X<0>);

// CHECK-LABEL: define weak_odr void @_Z2f0IJifdEEv1XIXsZT_EJDpRT_EE
template void f0<int, float, double>(X<3, int&, float&, double&>);

// Mangling for template argument packs
template<typename ...Types> void f1() {}
// CHECK-LABEL: define weak_odr void @_Z2f1IJEEvv
template void f1<>();
// CHECK-LABEL: define weak_odr void @_Z2f1IJiEEvv
template void f1<int>();
// CHECK-LABEL: define weak_odr void @_Z2f1IJifEEvv
template void f1<int, float>();

// Mangling function parameter packs
template<typename ...Types> void f2(Types...) {}
// CHECK-LABEL: define weak_odr void @_Z2f2IJEEvDpT_
template void f2<>();
// CHECK-LABEL: define weak_odr void @_Z2f2IJiEEvDpT_
template void f2<int>(int);
// CHECK-LABEL: define weak_odr void @_Z2f2IJifEEvDpT_
template void f2<int, float>(int, float);

// Mangling non-trivial function parameter packs
template<typename ...Types> void f3(const Types *...) {}
// CHECK-LABEL: define weak_odr void @_Z2f3IJEEvDpPKT_
template void f3<>();
// CHECK-LABEL: define weak_odr void @_Z2f3IJiEEvDpPKT_
template void f3<int>(const int*);
// CHECK-LABEL: define weak_odr void @_Z2f3IJifEEvDpPKT_
template void f3<int, float>(const int*, const float*);

// Mangling of type pack expansions in a template argument
template<typename ...Types> tuple<Types...> f4() { return {}; }
// CHECK-LABEL: define weak_odr void @_Z2f4IJifdEE5tupleIJDpT_EEv
template tuple<int, float, double> f4();

// Mangling of type pack expansions in a function type
template<typename R, typename ...ArgTypes> identity<R(ArgTypes...)> f5() {}
// CHECK-LABEL: define weak_odr void @_Z2f5IiJifdEE8identityIFT_DpT0_EEv
template identity<int(int, float, double)> f5();

// Mangling of non-type template argument expansions
template<int ...Values> int_tuple<Values...> f6() {}
// CHECK-LABEL: define weak_odr void @_Z2f6IJLi1ELi2ELi3EEE9int_tupleIJXspT_EEEv
template int_tuple<1, 2, 3> f6();

// Mangling of template template argument expansions
template<template<typename> class ...Templates> 
template_tuple<Templates...> f7() {}
// CHECK-LABEL: define weak_odr void @_Z2f7IJ8identity13add_referenceEE14template_tupleIJDpT_EEv
template template_tuple<identity, add_reference> f7();

template<typename T, typename ...U> void f8(ArrayOfN<int, U..., T, typename U::type...>&) {}
// CHECK-LABEL: define weak_odr void @_Z2f8IiJ8identityIiES0_IfEEEvRAsPiDpT0_T_DpNS3_4typeEE_i
template void f8<int, identity<int>, identity<float>>(int (&)[6]);

template<typename ...T> void f10(ArrayOfN<T...> &) {}
// FIXME: This is wrong; should be @_Z3f10IJifEEvRAsZT__i
// CHECK-LABEL: define weak_odr void @_Z3f10IJifEEvRAsPDpT_E_i
template void f10<int, float>(int (&)[2]);
