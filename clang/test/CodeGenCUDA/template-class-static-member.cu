// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck -check-prefix=DEV %s

// RUN: %clang_cc1 -triple x86_64-gnu-linux -std=c++11 \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck -check-prefix=HOST %s

// Negative tests.

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck -check-prefix=DEV-NEG %s

#include "Inputs/cuda.h"

template <class T>
class A {
    static int h_member;
    __device__ static int d_member;
    __constant__ static int c_member;
    __managed__ static int m_member;
    const static int const_member = 0;
};

template <class T>
int A<T>::h_member;

template <class T>
__device__ int A<T>::d_member;

template <class T>
__constant__ int A<T>::c_member;

template <class T>
__managed__ int A<T>::m_member;

template <class T>
const int A<T>::const_member;

template class A<int>;

//DEV-DAG: @_ZN1AIiE8d_memberE = internal addrspace(1) global i32 0, comdat, align 4
//DEV-DAG: @_ZN1AIiE8c_memberE = internal addrspace(4) global i32 0, comdat, align 4
//DEV-DAG: @_ZN1AIiE8m_memberE = internal addrspace(1) externally_initialized global ptr addrspace(1) null
//DEV-DAG: @_ZN1AIiE12const_memberE = internal addrspace(4) constant i32 0, comdat, align 4
//DEV-NEG-NOT: @_ZN1AIiE8h_memberE

//HOST-DAG: @_ZN1AIiE8h_memberE = weak_odr global i32 0, comdat, align 4
//HOST-DAG: @_ZN1AIiE8d_memberE = internal global i32 undef, comdat, align 4
//HOST-DAG: @_ZN1AIiE8c_memberE = internal global i32 undef, comdat, align 4
//HOST-DAG: @_ZN1AIiE8m_memberE = internal externally_initialized global ptr null
//HOST-DAG: @_ZN1AIiE12const_memberE = weak_odr constant i32 0, comdat, align 4
