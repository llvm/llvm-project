// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

// PR48030

template<typename T> struct TLS { static thread_local T *mData; };
inline decltype(nullptr) non_constant_initializer() { return nullptr; }
template<typename T> thread_local T *TLS<T>::mData = non_constant_initializer();
struct S {};
S *current() { return TLS<S>::mData; };

// CHECK-DAG: @_ZN3TLSI1SE5mDataE = linkonce_odr thread_local global {{.*}}, comdat,
// CHECK-DAG: @_ZGVN3TLSI1SE5mDataE = linkonce_odr thread_local global {{.*}}, comdat($_ZN3TLSI1SE5mDataE),
// CHECK-DAG: @_ZTHN3TLSI1SE5mDataE = linkonce_odr alias {{.*}} @__cxx_global_var_init

// CHECK-LABEL: define {{.*}} @_Z7currentv()
// CHECK: call {{.*}} @_ZTWN3TLSI1SE5mDataE()

// CHECK-LABEL: define weak_odr hidden {{.*}} @_ZTWN3TLSI1SE5mDataE() {{.*}} comdat {
// CHECK: call void @_ZTHN3TLSI1SE5mDataE()
// CHECK: [[TLSmData_ADDR:%[^ ]+]] = call ptr @llvm.threadlocal.address.p0(ptr @_ZN3TLSI1SE5mDataE)
// CHECK: ret {{.*}} [[TLSmData_ADDR]]

// Unlike for a global, the global initialization function must not be in a
// COMDAT with the variable, because it is referenced from the _ZTH function
// which is outside that COMDAT.
//
// CHECK-NOT: define {{.*}} @__cxx_global_var_init{{.*}}comdat
