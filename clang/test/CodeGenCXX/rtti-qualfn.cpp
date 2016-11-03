// RUN: %clang_cc1 -std=c++1z -mqualified-function-type-info -I%S %s -triple x86_64-linux-gnu -emit-llvm -o - -fcxx-exceptions | FileCheck %s

#include "typeinfo"

struct A {};

// CHECK-DAG: @_ZTIKFvvE = [[QFTI:linkonce_odr constant { i8\*, i8\*, i8\*, i32 } { i8\* bitcast \(i8\*\* getelementptr inbounds \(i8\*, i8\*\* @_ZTVN10__cxxabiv130__qualified_function_type_infoE, i64 2\) to i8\*\),]] i8* getelementptr inbounds ([6 x i8], [6 x i8]* @_ZTSKFvvE, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTIFvvE to i8*), i32 1 }, comdat
// CHECK-DAG: @_ZTIM1AKFvvE = [[PMFTI:linkonce_odr constant { i8\*, i8\*, i32, i8\*, i8\* } { i8\* bitcast \(i8\*\* getelementptr inbounds \(i8\*, i8\*\* @_ZTVN10__cxxabiv129__pointer_to_member_type_infoE, i64 2\) to i8\*\),]] i8* getelementptr inbounds ([9 x i8], [9 x i8]* @_ZTSM1AKFvvE, i32 0, i32 0), i32 0, i8* bitcast ({ i8*, i8*, i8*, i32 }* @_ZTIKFvvE to i8*), i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*) }, comdat
auto &ti_const = typeid(void (A::*)() const);

// CHECK-DAG: @_ZTIVFvvE = [[QFTI]] {{.*}} @_ZTIFvvE {{.*}}, i32 2 }, comdat
// CHECK-DAG: @_ZTIM1AVFvvE = [[PMFTI]] {{.*}}), i32 0, {{.*}} @_ZTIVFvvE
auto &ti_volatile = typeid(void (A::*)() volatile);

// CHECK-DAG: @_ZTIrFvvE = [[QFTI]] {{.*}} @_ZTIFvvE {{.*}}, i32 4 }, comdat
// CHECK-DAG: @_ZTIM1ArFvvE = [[PMFTI]] {{.*}}), i32 0, {{.*}} @_ZTIrFvvE
auto &ti_restrict = typeid(void (A::*)() __restrict);

// CHECK-DAG: @_ZTIFvvRE = [[QFTI]] {{.*}} @_ZTIFvvE {{.*}}, i32 8 }, comdat
// CHECK-DAG: @_ZTIM1AFvvRE = [[PMFTI]] {{.*}}), i32 0, {{.*}} @_ZTIFvvRE
auto &ti_lref = typeid(void (A::*)() &);

// CHECK-DAG: @_ZTIFvvOE = [[QFTI]] {{.*}} @_ZTIFvvE {{.*}}, i32 16 }, comdat
// CHECK-DAG: @_ZTIM1AFvvOE = [[PMFTI]] {{.*}}), i32 0, {{.*}} @_ZTIFvvOE
auto &ti_rref = typeid(void (A::*)() &&);

// CHECK-DAG: @_ZTIDoFvvE = [[QFTI]] {{.*}} @_ZTIFvvE {{.*}}, i32 32 }, comdat
// CHECK-DAG: @_ZTIM1ADoFvvE = [[PMFTI]] {{.*}}), i32 0, {{.*}} @_ZTIDoFvvE
auto &ti_noexcept = typeid(void (A::*)() noexcept);

//auto &ti_txsafe = typeid(void (A::*)() transaction_safe);

// FIXME: Produce the typeinfo for a noreturn function type here?
// CHECK-DAG: @_ZTIM1AFvvE = [[PMFTI]] {{.*}}), i32 0, {{.*}} @_ZTIFvvE
auto &ti_noreturn = typeid(void __attribute__((noreturn)) (A::*)());

// CHECK-DAG: @_ZTIrVKDoFvvRE = [[QFTI]] {{.*}} @_ZTIFvvE {{.*}}, i32 47 }, comdat
// CHECK-DAG: @_ZTIM1ArVKDoFvvRE = [[PMFTI]] {{.*}}), i32 0, {{.*}} @_ZTIrVKDoFvvRE
auto &ti_rainbow = typeid(void (A::*)() const volatile __restrict & noexcept);

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
