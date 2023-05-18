// RUN: %clang_cc1 -std=c++1z -I%S %s -triple x86_64-linux-gnu -emit-llvm -o - -fcxx-exceptions | FileCheck %s

#include "typeinfo"

struct A {};

// CHECK-DAG: @_ZTIFvvE = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__function_type_infoE, i64 2), ptr @_ZTSFvvE }, comdat
// CHECK-DAG: @_ZTIPDoFvvE = linkonce_odr constant { ptr, ptr, i32, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2), ptr @_ZTSPDoFvvE, i32 64, ptr @_ZTIFvvE }, comdat
auto &ti_noexcept_ptr = typeid(void (A::*)() noexcept);
// CHECK-DAG: @_ZTIM1ADoFvvE = linkonce_odr constant { ptr, ptr, i32, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 2), ptr @_ZTSM1ADoFvvE, i32 64, ptr @_ZTIFvvE, ptr @_ZTI1A }, comdat
auto &ti_noexcept_memptr = typeid(void (A::*)() noexcept);

// CHECK-LABEL: define{{.*}} void @_Z1fv(
__attribute__((noreturn)) void f() noexcept {
  // CHECK: call void @__cxa_throw({{.*}}@_ZTIPDoFvvE
  throw f;
}

// CHECK-LABEL: define{{.*}} void @_Z1gM1ADoFvvE(
void g(__attribute__((noreturn)) void (A::*p)() noexcept) {
  // CHECK: call void @__cxa_throw({{.*}}@_ZTIM1ADoFvvE
  throw p;
}
