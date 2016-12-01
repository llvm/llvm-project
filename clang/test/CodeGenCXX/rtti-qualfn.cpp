// RUN: %clang_cc1 -std=c++1z -I%S %s -triple x86_64-linux-gnu -emit-llvm -o - -fcxx-exceptions | FileCheck %s

#include "typeinfo"

struct A {};

// CHECK-DAG: @_ZTIFvvE = linkonce_odr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__function_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZTSFvvE, i32 0, i32 0) }, comdat
// CHECK-DAG: @_ZTIPDoFvvE = linkonce_odr constant { i8*, i8*, i32, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_ZTSPDoFvvE, i32 0, i32 0), i32 64, i8* bitcast ({ i8*, i8* }* @_ZTIFvvE to i8*) }, comdat
auto &ti_noexcept_ptr = typeid(void (A::*)() noexcept);
// CHECK-DAG: @_ZTIM1ADoFvvE = linkonce_odr constant { i8*, i8*, i32, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @_ZTSM1ADoFvvE, i32 0, i32 0), i32 64, i8* bitcast ({ i8*, i8* }* @_ZTIFvvE to i8*), i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*) }, comdat
auto &ti_noexcept_memptr = typeid(void (A::*)() noexcept);

// CHECK-LABEL: define void @_Z1fv(
__attribute__((noreturn)) void f() noexcept {
  // CHECK: call void @__cxa_throw({{.*}}@_ZTIPDoFvvE
  throw f;
}

// CHECK-LABEL: define void @_Z1gM1ADoFvvE(
void g(__attribute__((noreturn)) void (A::*p)() noexcept) {
  // CHECK: call void @__cxa_throw({{.*}}@_ZTIM1ADoFvvE
  throw p;
}
