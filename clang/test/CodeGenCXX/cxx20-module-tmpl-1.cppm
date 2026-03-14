// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

export module FOO;

class One;
class Two;

export template<typename T> struct TPL
{
void M (T *);
template<typename U> void N (T *, U*);
};

template<typename T>
void TPL<T>::M (T *) {}

template<typename T> template<typename U> void TPL<T>::N (T *, U*) {}

// CHECK-DAG: void @_ZNW3FOO3TPLIS_3OneE1MEPS1_(
template void TPL<One>::M (One *);
// CHECK-DAG: void @_ZNW3FOO3TPLIS_3OneE1NIS_3TwoEEvPS1_PT_(
template void TPL<One>::N<Two> (One *, Two *);

namespace NMS {
class One;
class Two;

export template<typename T> struct TPL
{
void M (T *);
template<typename U> void N (T *, U*);
};

template<typename T>
void TPL<T>::M (T *) {}

template<typename T> template<typename U> void TPL<T>::N (T *, U*) {}

// CHECK-DAG: void @_ZN3NMSW3FOO3TPLINS_S0_3OneEE1MEPS2_(
template void TPL<One>::M (One *);
// CHECK-DAG: void @_ZN3NMSW3FOO3TPLINS_S0_3OneEE1NINS_S0_3TwoEEEvPS2_PT_
template void TPL<One>::N<Two> (One *, Two *);
}
