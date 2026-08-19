// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 %s -triple %itanium_abi_triple -Wno-unused-value -std=c++11 -fclangir -emit-llvm -o - -std=c++11 | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 %s -triple %itanium_abi_triple -Wno-unused-value -std=c++11 -emit-llvm -o - -std=c++11 | FileCheck %s --check-prefixes=LLVM
namespace std {
struct type_info;
}

struct A {
  virtual ~A();
  operator bool();
};
struct B : A {};

void f1(A *x) { typeid(false, *x); }
// CIR-LABEL: cir.func{{.*}}@_Z2f1P1A
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z2f1P1A
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f2(bool b, A *x, A *y) { typeid(b ? *x : *y); }
// CIR-LABEL: cir.func{{.*}}@_Z2f2bP1AS0_
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z2f2bP1AS0_
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f3(bool b, A *x, A &y) { typeid(b ? *x : y); }
// CIR-LABEL: cir.func{{.*}}@_Z2f3bP1ARS_
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z2f3bP1ARS_
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f4(bool b, A &x, A *y) { typeid(b ? x : *y); }
// CIR-LABEL: cir.func{{.*}}@_Z2f4bR1APS_
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z2f4bR1APS_
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f5(volatile A *x) { typeid(*x); }
// CIR-LABEL: cir.func{{.*}}@_Z2f5PV1A
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z2f5PV1A
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f6(A *x) { typeid((B &)*(B *)x); }
// CIR-LABEL: cir.func{{.*}}@_Z2f6P1A
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z2f6P1A
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f7(A *x) { typeid((*x)); }
// CIR-LABEL: cir.func{{.*}}@_Z2f7P1A
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z2f7P1A
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f8(A *x) { typeid(x[0]); }
// CIR-LABEL: cir.func{{.*}}@_Z2f8P1A
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z2f8P1A
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f9(A *x) { typeid(0[x]); }
// CIR-LABEL: cir.func{{.*}}@_Z2f9P1A
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z2f9P1A
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f10(A *x, A *y) { typeid(*y ?: *x); }
// CIR-LABEL: cir.func{{.*}}@_Z3f10P1AS0_
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z3f10P1AS0_
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f11(A *x, A &y) { typeid(*x ?: y); }
// CIR-LABEL: cir.func{{.*}}@_Z3f11P1ARS_
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z3f11P1ARS_
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f12(A &x, A *y) { typeid(x ?: *y); }
// CIR-LABEL: cir.func{{.*}}@_Z3f12R1APS_
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %{{.*}}, %[[NULL]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z3f12R1APS_
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1

void f13(A &x, A &y) { typeid(x ?: y); }
// CIR-LABEL: cir.func{{.*}}@_Z3f13R1AS0_
// CIR-NOT: @__cxa_bad_typeid()
// LLVM-LABEL: define {{.*}}void @_Z3f13R1AS0_
// LLVM-NOT:   icmp eq {{.*}}, null

void f14(A *x) { typeid((const A &)(A)*x); }
// CIR-LABEL: cir.func{{.*}}@_Z3f14P1A
// CIR-NOT: @__cxa_bad_typeid()
// LLVM-LABEL: define {{.*}}void @_Z3f14P1A
// LLVM-NOT:   icmp eq {{.*}}, null

void f15(A *x) { typeid((A &&)*(A *)nullptr); }
// CIR-LABEL: cir.func{{.*}}@_Z3f15P1A
// In this example, it only passes classic codegen because the icmp doesn't
// happen, it just does a branch on 'true' thanks to constant folding.  So we're
// consistant with classic codegen.
// CIR: %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[NULL2:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT: %[[CMP:.*]] = cir.cmp eq %[[NULL]], %[[NULL2]]
// CIR-NEXT: cir.if %[[CMP]]
// CIR-NEXT: cir.call @__cxa_bad_typeid() {noreturn} : () -> ()
// CIR-NEXT: cir.unreachable
// LLVM-LABEL: define {{.*}}void @_Z3f15P1A
// LLVM-NOT:   icmp eq {{.*}}, null
